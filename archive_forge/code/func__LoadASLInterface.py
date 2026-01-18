from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def _LoadASLInterface(libname):
    ASLib = ctypes.cdll.LoadLibrary(libname)
    array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
    array_1d_int = np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')
    try:
        ASLib.EXTERNAL_AmplInterface_version.argtypes = None
        ASLib.EXTERNAL_AmplInterface_version.restype = ctypes.c_int
        interface_version = ASLib.EXTERNAL_AmplInterface_version()
    except AttributeError:
        interface_version = 1
    if interface_version >= 3:
        ASLib.EXTERNAL_get_asl_date.argtypes = []
        ASLib.EXTERNAL_get_asl_date.restype = ctypes.c_long
    ASLib.EXTERNAL_AmplInterface_new.argtypes = [ctypes.c_char_p]
    ASLib.EXTERNAL_AmplInterface_new.restype = ctypes.c_void_p
    if interface_version >= 2:
        ASLib.EXTERNAL_AmplInterface_new_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    else:
        ASLib.EXTERNAL_AmplInterface_new_file.argtypes = [ctypes.c_char_p]
    ASLib.EXTERNAL_AmplInterface_new_file.restype = ctypes.c_void_p
    ASLib.EXTERNAL_AmplInterface_n_vars.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_n_vars.restype = ctypes.c_int
    ASLib.EXTERNAL_AmplInterface_n_constraints.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_n_constraints.restype = ctypes.c_int
    ASLib.EXTERNAL_AmplInterface_nnz_jac_g.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_nnz_jac_g.restype = ctypes.c_int
    ASLib.EXTERNAL_AmplInterface_nnz_hessian_lag.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_nnz_hessian_lag.restype = ctypes.c_int
    ASLib.EXTERNAL_AmplInterface_x_lower_bounds.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_x_lower_bounds.restype = None
    ASLib.EXTERNAL_AmplInterface_x_upper_bounds.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_x_upper_bounds.restype = None
    ASLib.EXTERNAL_AmplInterface_g_lower_bounds.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_g_lower_bounds.restype = None
    ASLib.EXTERNAL_AmplInterface_g_upper_bounds.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_g_upper_bounds.restype = None
    ASLib.EXTERNAL_AmplInterface_get_init_x.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_get_init_x.restype = None
    ASLib.EXTERNAL_AmplInterface_get_init_multipliers.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_get_init_multipliers.restype = None
    ASLib.EXTERNAL_AmplInterface_eval_f.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    ASLib.EXTERNAL_AmplInterface_eval_f.restype = ctypes.c_bool
    ASLib.EXTERNAL_AmplInterface_eval_deriv_f.argtypes = [ctypes.c_void_p, array_1d_double, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_eval_deriv_f.restype = ctypes.c_bool
    ASLib.EXTERNAL_AmplInterface_struct_jac_g.argtypes = [ctypes.c_void_p, array_1d_int, array_1d_int, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_struct_jac_g.restype = None
    ASLib.EXTERNAL_AmplInterface_struct_hes_lag.argtypes = [ctypes.c_void_p, array_1d_int, array_1d_int, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_struct_hes_lag.restype = None
    ASLib.EXTERNAL_AmplInterface_eval_g.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_eval_g.restype = ctypes.c_bool
    ASLib.EXTERNAL_AmplInterface_eval_jac_g.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_eval_jac_g.restype = ctypes.c_bool
    try:
        ASLib.EXTERNAL_AmplInterface_dummy.argtypes = [ctypes.c_void_p]
        ASLib.EXTERNAL_AmplInterface_dummy.restype = None
        ASLib.EXTERNAL_AmplInterface_eval_hes_lag.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int, array_1d_double, ctypes.c_int, array_1d_double, ctypes.c_int, ctypes.c_double]
        ASLib.EXTERNAL_AmplInterface_eval_hes_lag.restype = ctypes.c_bool
    except Exception:
        ASLib.EXTERNAL_AmplInterface_eval_hes_lag.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_int, array_1d_double, ctypes.c_int, array_1d_double, ctypes.c_int]
        ASLib.EXTERNAL_AmplInterface_eval_hes_lag.restype = ctypes.c_bool
        interface_version = 0
    ASLib.EXTERNAL_AmplInterface_finalize_solution.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, array_1d_double, ctypes.c_int, array_1d_double, ctypes.c_int]
    ASLib.EXTERNAL_AmplInterface_finalize_solution.restype = None
    ASLib.EXTERNAL_AmplInterface_free_memory.argtypes = [ctypes.c_void_p]
    ASLib.EXTERNAL_AmplInterface_free_memory.restype = None
    if CURRENT_INTERFACE_VERSION != interface_version:
        logger.warning('The current pynumero_ASL library is version=%s, but found version=%s.  Please recompile / update your pynumero_ASL library.' % (CURRENT_INTERFACE_VERSION, interface_version))
    return (ASLib, interface_version)