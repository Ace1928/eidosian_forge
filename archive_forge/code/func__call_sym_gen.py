import logging
import warnings
import numpy as np
from .. import context as ctx
from ..initializer import Uniform
from .. import ndarray as nd
from .. import symbol as sym
from .base_module import BaseModule, _check_input_names
from .module import Module
from ..model import load_params
from ..name import NameManager
def _call_sym_gen(self, *args, **kwargs):
    with NameManager():
        return self._sym_gen(*args, **kwargs)