from typing import Dict, Tuple
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.nn import Module
@compatibility(is_backward_compatible=False)
class HolderModule(Module):
    """
    HolderModule is used to copy all the attributes from original module to submodules
    that uses the attributes
    """

    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            self.add_module(k, v)