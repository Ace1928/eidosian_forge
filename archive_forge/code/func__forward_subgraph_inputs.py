import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def _forward_subgraph_inputs(self, subgraph: List[fx.Node], graph: fx.Graph, erase_node: bool) -> int:
    """Turn the inputs of a subgraph into the extra output of the entire graph.

        If ``erase_node`` is True, the subgraph will be erased from the graph -- essentially forward the inputs
        of the subgraph to the output of the graph.
        """
    output = get_output(graph)
    inputs = []
    all_nodes: Set[fx.Node] = set(subgraph)
    for node in subgraph:
        node_inputs = pytree.arg_tree_leaves(*node.args, **node.kwargs)
        for _input in node_inputs:
            if not isinstance(_input, fx.Node):
                continue
            if _input in all_nodes:
                continue
            inputs.append(_input)
    if erase_node:
        erased = set()
        for node in reversed(subgraph):
            if len(node.users) == 1:
                key = next(iter(node.users.keys()))
                if key == output:
                    flatten_args, spec = tree_flatten((output.args, output.kwargs))
                    if node not in flatten_args:
                        node.users.clear()
                    elif str(node.target).startswith('aten.copy_'):
                        for i in range(len(flatten_args)):
                            if flatten_args[i] == node:
                                flatten_args[i] = node.args[0]
                    else:
                        raise RuntimeError(f'IterGraph does not how to forward the output of {node}')
                    output.args, output.kwargs = tree_unflatten(flatten_args, spec)
            for user in list(node.users.keys()):
                if user in erased:
                    node.users.pop(user)
            if node.users:
                raise RuntimeError(f'IterGraph has not supported moving the nodes that produce users output result. Error node: {node}.')
            self._fx_graph_call(graph, 'erase_node', node)
            erased.add(node)
    if self.num_extra_output:
        cast(List[fx.Node], output.args[0][-1]).extend(inputs)
        new_output = output.args[0]
    elif isinstance(graph._codegen, _PyTreeCodeGen):
        codegen = graph._codegen
        new_output = list(output.args[0])
        new_output.append(inputs)
        assert codegen.pytree_info.out_spec is not None
        original_tree_out = tree_unflatten(cast(List[Any], output.args[0]), codegen.pytree_info.out_spec)
        _, out_spec = tree_flatten((original_tree_out, None))
        codegen.pytree_info = codegen.pytree_info._replace(out_spec=out_spec)
    else:
        new_output = (output.args[0], inputs)
    self._fx_graph_call(graph, 'erase_node', output)
    self._fx_graph_call(graph, 'output', new_output)
    logger.info('Extended outputs from the subgraph inputs: %s', str(inputs))
    return len(inputs)