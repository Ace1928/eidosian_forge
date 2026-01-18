import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
def _inject_new_class(module: Module) -> None:
    """Set up a module to be parametrized.

    This works by substituting the class of the module by a class
    that extends it to be able to inject a property

    Args:
        module (nn.Module): module into which to inject the property
    """
    cls = module.__class__

    def default_deepcopy(self, memo):
        obj = memo.get(id(self), None)
        if obj is not None:
            return obj
        replica = self.__new__(self.__class__)
        memo[id(self)] = replica
        replica.__dict__ = deepcopy(self.__dict__, memo)
        slots_to_save = copyreg._slotnames(self.__class__)
        for slot in slots_to_save:
            if hasattr(self, slot):
                setattr(replica, slot, deepcopy(getattr(self, slot), memo))
        return replica

    def getstate(self):
        raise RuntimeError('Serialization of parametrized modules is only supported through state_dict(). See:\nhttps://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training')
    dct = {'__getstate__': getstate}
    if not hasattr(cls, '__deepcopy__'):
        dct['__deepcopy__'] = default_deepcopy
    param_cls = type(f'Parametrized{cls.__name__}', (cls,), dct)
    module.__class__ = param_cls