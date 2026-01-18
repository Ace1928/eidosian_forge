import unittest
from contextlib import contextmanager
from functools import cached_property
from numba import njit
from numba.core import errors, cpu, typing
from numba.core.descriptors import TargetDescriptor
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.retarget import BasicRetarget
from numba.core.extending import overload
from numba.core.target_extension import (
class CustomCPURetarget(BasicRetarget):

    @property
    def output_target(self):
        return CUSTOM_TARGET

    def compile_retarget(self, cpu_disp):
        kernel = njit(_target=CUSTOM_TARGET)(cpu_disp.py_func)
        return kernel