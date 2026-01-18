import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Union, Dict, Any, Set
class ObservedStandaloneGraphModule(ObservedGraphModule):

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, preserved_attr_names: Set[str]):
        preserved_attr_names = preserved_attr_names.union({'_standalone_module_input_quantized_idxs', '_standalone_module_output_quantized_idxs'})
        super().__init__(root, graph, preserved_attr_names)

    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedStandaloneGraphModule(fake_mod, copy.deepcopy(self.graph), copy.deepcopy(self.preserved_attr_names))