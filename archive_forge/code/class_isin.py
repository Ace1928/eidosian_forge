import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
class isin:

    def __contains__(self, item):
        for x in self:
            if seq(item, x):
                return True
        return False

    def index(self, item):
        for i, x in enumerate(self):
            if seq(item, x):
                return i
        raise ValueError