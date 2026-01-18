import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable
class PlacementClassVariable(DistributedVariable):

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_placement_type(value):
        if not DistributedVariable.is_available():
            return False
        from torch.distributed._tensor.placement_types import Placement
        return type(value) is type and issubclass(value, Placement)

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if inspect.getattr_static(self.value, '__new__', None) in (object.__new__,) and self.source:
            new_obj = object.__new__(self.value)
            var = PlacementVariable(new_obj)
            if inspect.getattr_static(self.value, '__init__', None):
                var.call_method(tx, '__init__', args, kwargs)
                return var
        return super().call_function(tx, args, kwargs)