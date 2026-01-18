import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable
class DeviceMeshVariable(DistributedVariable):

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_device_mesh(value):
        if not DistributedVariable.is_available():
            return False
        from torch.distributed.device_mesh import DeviceMesh
        return istype(value, DeviceMesh)

    def as_python_constant(self):
        return self.value

    def var_getattr(self, tx, name: str) -> VariableTracker:
        if name == 'ndim':
            return ConstantVariable.create(self.value.ndim)
        return super().var_getattr(tx, name)