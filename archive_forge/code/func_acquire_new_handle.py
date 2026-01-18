import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def acquire_new_handle(self):
    self.__class__.active_global_handle += 1
    self.handle = self.__class__.active_global_handle