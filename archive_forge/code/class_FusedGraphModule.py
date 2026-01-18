import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Union, Dict, Any, Set
class FusedGraphModule(GraphModule):

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {attr: getattr(root, attr) for attr in self.preserved_attr_names if hasattr(root, attr)}
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return FusedGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))