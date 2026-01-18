import functools
from typing import Dict, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from ..pattern_matcher import fwd_only, register_replacement
class NumpyCompatNormalization:
    numpy_compat: Dict[str, Tuple[str, ...]] = {'dim': ('axis',), 'keepdim': ('keepdims',), 'input': ('x', 'a', 'x1'), 'other': ('x2',)}
    inverse_mapping: Dict[str, str]
    cache: Dict['torch.fx.graph.Target', Set[str]]

    def __init__(self):
        self.cache = {}
        self.inverse_mapping = {}
        for actual_kwarg, numpy_kwargs in self.numpy_compat.items():
            for numpy_kwarg in numpy_kwargs:
                assert numpy_kwarg not in self.inverse_mapping
                self.inverse_mapping[numpy_kwarg] = actual_kwarg

    def __call__(self, graph: torch.fx.Graph):
        for node in graph.nodes:
            if node.op != 'call_function':
                continue
            if isinstance(node.target, (OpOverload, OpOverloadPacket)):
                continue
            kwargs = node.kwargs
            if node.target in self.cache:
                replaceable_kwargs = self.cache[node.target]
            else:
                signatures = torch.fx.operator_schemas.get_signature_for_torch_op(node.target)
                signatures = () if signatures is None else signatures
                replaceable_kwargs = set()
                for sig in signatures:
                    for param_name in sig.parameters.keys():
                        if param_name in self.numpy_compat:
                            replaceable_kwargs.update(self.numpy_compat[param_name])
                self.cache[node.target] = replaceable_kwargs
            if not replaceable_kwargs:
                continue
            new_kwargs = {}
            kwargs_changed = False
            for k, v in kwargs.items():
                if k in replaceable_kwargs:
                    kwargs_changed = True
                    new_kwargs[self.inverse_mapping[k]] = v
                else:
                    new_kwargs[k] = v
            if kwargs_changed:
                node.kwargs = torch.fx.immutable_collections.immutable_dict(new_kwargs)
                counters['inductor']['numpy_compat_normalization'] += 1