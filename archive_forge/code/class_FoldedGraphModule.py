import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module
class FoldedGraphModule(torch.fx.GraphModule):
    """
    FoldedGraphModule is a GraphModule which also contains another
    `const_subgraph_module` representing a subgraph which has all const attr
    inputs and which can be run once before running the main standard
    `graph`. The `const_output_names` are the ordered list names of attrs which
    represent what each respective output from the const_subgraph should be set
    on which attrs.
    """

    def __init__(self, root: torch.nn.Module, graph: torch.fx.Graph, const_subgraph: Optional[torch.fx.Graph]=None, fx_const_folded_attrs_name: Optional[str]=None, device_for_folded_attrs: str='cuda'):
        super().__init__(root, graph)
        self.const_subgraph_module = None if const_subgraph is None else torch.fx.GraphModule(root, const_subgraph)
        self.has_folding_been_run = False
        self.fx_const_folded_attrs_name = fx_const_folded_attrs_name
        self.device_for_folded_attrs = device_for_folded_attrs

    def __call__(self, *args, **kwargs):
        if not self.has_folding_been_run:
            self.run_folding()
        return super().__call__(*args)

    def run_folding(self):
        if self.const_subgraph_module is None or self.fx_const_folded_attrs_name is None:
            return
        assert not self.has_folding_been_run
        self.has_folding_been_run = True
        folded_attrs = self.const_subgraph_module()

        def _create_param(i):
            return torch.nn.Parameter(i if not isinstance(i, int) else torch.Tensor([i]).to(device=self.device_for_folded_attrs), requires_grad=i.requires_grad if isinstance(i, torch.Tensor) else False)
        params = torch.nn.ParameterList([_create_param(i) for i in folded_attrs]) if isinstance(folded_attrs, tuple) else _create_param(folded_attrs)
        setattr(self, self.fx_const_folded_attrs_name, params)