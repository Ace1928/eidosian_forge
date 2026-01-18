from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import cbc_path, pulp_cbc_path, coinMP_path, devnull, operating_system
import os
from .. import constants
from tempfile import mktemp
import ctypes
import warnings
def COINMP_DLL_load_dll(path):
    """
    function that loads the DLL useful for debugging installation problems
    """
    if os.name == 'nt':
        lib = ctypes.windll.LoadLibrary(str(path[-1]))
    else:
        mode = ctypes.RTLD_GLOBAL
        for libpath in path[:-1]:
            ctypes.CDLL(libpath, mode=mode)
        lib = ctypes.CDLL(path[-1], mode=mode)
    return lib