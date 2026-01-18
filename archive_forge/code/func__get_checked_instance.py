import warnings
from typing import Any, Dict, Optional, Tuple
import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from torch.types import _size
def _get_checked_instance(self, cls, _instance=None):
    if _instance is None and type(self).__init__ != cls.__init__:
        raise NotImplementedError(f'Subclass {self.__class__.__name__} of {cls.__name__} that defines a custom __init__ method must also define a custom .expand() method.')
    return self.__new__(type(self)) if _instance is None else _instance