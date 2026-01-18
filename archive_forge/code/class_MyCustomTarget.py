import numpy as np
from numba import njit
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.target_extension import (
from numba.core.retarget import BasicRetarget
class MyCustomTarget(BasicRetarget):

    @property
    def output_target(self):
        return 'CustomCPU'

    def compile_retarget(self, cpu_disp):
        kernel = njit(_target='CustomCPU', parallel=True)(cpu_disp.py_func)
        return kernel