from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
class ReplaceGetAttrWithPlaceholder(_pass.Transform):
    """Replace get_attr with placeholder.

    The parameters and buffers accessed by the original get_attr are returned;
    they are useful when creating random inputs for the modified graph_module.
    """
    _replaced_attrs: Optional[Tuple[torch.Tensor, ...]]

    @property
    def replaced_attrs(self) -> Tuple[torch.Tensor, ...]:
        """The list of replaced weight tensors."""
        assert self._replaced_attrs is not None, 'Must run ReplaceGetAttrWithPlaceholder first'
        return self._replaced_attrs

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        graph_module = self.module
        graph = graph_module.graph
        replaced_attrs: List[torch.Tensor] = []
        for node in graph.nodes:
            if node.op == 'get_attr':
                replaced_attr: Optional[torch.Tensor] = None
                try:
                    replaced_attr = graph_module.get_parameter(node.target)
                except AttributeError:
                    replaced_attr = graph_module.get_buffer(node.target)
                node.op = 'placeholder'
                node.target = node.target.replace('.', '_')
                node.args = (None,)
                replaced_attrs.append(replaced_attr)
        self._replaced_attrs = tuple(replaced_attrs)
        return graph_module