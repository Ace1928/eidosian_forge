import functools
import itertools
import logging
import operator
from collections import Counter, defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Union
from sympy import Expr
import torch
import torch._inductor as inductor
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp import register_decomposition
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional
from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype
from torch._utils_internal import print_graph
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.fx.immutable_collections import immutable_dict
from .. import config, inductor_prims, ir, pattern_matcher
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage
from ..lowering import (
from ..pattern_matcher import (
from ..utils import decode_device, is_pointwise_use
from ..virtualized import V
from .group_batch_fusion import group_batch_fusion_passes
class ConstructorMoverPass:

    def __init__(self, target: str, allow_outputs: bool=False) -> None:
        """
        Move constructors from cpu to the target_device.

        Sweeps through the module, looking for constructor nodes that can be moved
        to the target_device.

        A constructor node can be moved to the target_device iff all of its users
        can also be moved (tested by cannot_be_moved). Otherwise, all dependent
        constructor nodes won't be moved.

        - target: target device type
        - allow_outputs: allow outputs to be moved
        """
        self.target = target
        self.allow_outputs = allow_outputs
        assert isinstance(target, str), f'target should be a string representing the device type. Got: {type(target).__name__}'

    def allow_cpu_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node that returns a tensor on the target device may have
        cpu tensors as input.
        """
        return node.target in (torch.ops.aten.index.Tensor, torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default, torch.ops.aten.copy.default, torch.ops.aten.copy_.default, torch.ops.aten.slice_scatter.default)

    def cannot_be_moved(self, node: fx.Node) -> bool:
        """
        Returns whether a node can be moved to the target device.

        If this function returns False, it means that this node and all of its users
        won't be moved into the target device.
        """
        if node.target == 'output':
            return not self.allow_outputs
        if not (isinstance(node.target, torch._ops.OpOverload) and node.target.namespace in ('prims', 'aten')):
            return True
        return False

    def get_node_device(self, node: fx.Node) -> Optional[torch.device]:
        """
        Get the device of a node.
        """
        ten = node.meta.get('val')
        return None if not isinstance(ten, torch.Tensor) else ten.device

    def get_cpu_indeg_count(self, graph: fx.Graph) -> Dict[fx.Node, int]:
        """
        Get the number of cpu inputs to a node
        """
        cpu_indeg: Dict[fx.Node, int] = Counter()
        for node in graph.nodes:
            cpu_count = 0

            def add_cpu_inp(node):
                nonlocal cpu_count
                device = self.get_node_device(node)
                cpu_count += device is not None and device.type == 'cpu'
            pytree.tree_map_only(fx.Node, add_cpu_inp, (node.args, node.kwargs))
            if cpu_count:
                cpu_indeg[node] = cpu_count
        return cpu_indeg

    def __call__(self, graph: fx.Graph) -> None:
        target_devices = set()
        constructors = []
        for node in graph.nodes:
            device = self.get_node_device(node)
            if device and device.type == self.target:
                target_devices.add(device)
            if not (isinstance(node.target, torch._ops.OpOverload) and node.target.namespace in ('prims', 'aten')):
                continue
            if not torch._subclasses.fake_tensor._is_tensor_constructor(node.target):
                continue
            if not node.kwargs.get('device') == torch.device('cpu'):
                continue
            constructors.append(node)
        if not constructors or len(target_devices) != 1:
            return
        movable_constructors = self.find_movable_constructors(graph, constructors)
        for node in movable_constructors:
            kwargs = node.kwargs.copy()
            kwargs['device'] = next(iter(target_devices))
            node.kwargs = kwargs

    def find_movable_constructors(self, graph: fx.Graph, constructors: List[fx.Node]) -> Set[fx.Node]:
        """
        Starting from the cpu constructors, iterate through the graph and test that all of their
        downstream uses can safely be moved to cpu.
        """
        cpu_indeg: Dict[fx.Node, int] = self.get_cpu_indeg_count(graph)
        cannot_move_to_cuda: Set[fx.Node] = set()
        constructor_dependencies: Dict[fx.Node, Set[fx.Node]] = defaultdict(set)
        equal_constructor_sets: Dict[fx.Node, Set[fx.Node]] = {c: {c} for c in constructors}

        def make_dependencies_equivalent(set1: Set[fx.Node], set2: Set[fx.Node]) -> Set[fx.Node]:
            set1.update(set2)
            for obj in set1:
                equal_constructor_sets[obj] = set1
            return set1
        queue: List[fx.Node] = list(constructors)
        for c in queue:
            constructor_dependencies[c].add(c)
        while queue:
            node = queue.pop()
            dependencies = constructor_dependencies[node]
            for user in node.users:
                if self.cannot_be_moved(user):
                    cannot_move_to_cuda.update(dependencies)
                    break
                node_device = self.get_node_device(user)
                if self.allow_cpu_device(user) and node_device and (node_device.type == self.target):
                    del cpu_indeg[user]
                else:
                    cpu_indeg[user] -= 1
                    if cpu_indeg[user] == 0:
                        del cpu_indeg[user]
                        queue.append(user)
                unioned_set = make_dependencies_equivalent(dependencies, constructor_dependencies[user])
                constructor_dependencies[user] = unioned_set
        for node in cpu_indeg:
            if constructor_dependencies[node]:
                cannot_move_to_cuda.update(constructor_dependencies[node])
        all_cannot_move_to_cuda = cannot_move_to_cuda.copy()
        for constructor in cannot_move_to_cuda:
            all_cannot_move_to_cuda.update(equal_constructor_sets[constructor])
        return set(constructors) - all_cannot_move_to_cuda