import logging
import types
import weakref
from pyomo.common.pyomo_typing import overload
from ctypes import (
from pyomo.common.autoslots import AutoSlots
from pyomo.common.fileutils import find_library
from pyomo.core.expr.numvalue import (
import pyomo.core.expr as EXPR
from pyomo.core.base.component import Component
from pyomo.core.base.units_container import units
class _ARGLIST(Structure):
    """Mock up the arglist structure from AMPL's funcadd.h

    This data structure is populated by AMPL when calling external
    functions (for both passing in the function arguments and retrieving
    the derivative/Hessian information.
    """
    _fields_ = [('n', c_int), ('nr', c_int), ('at', POINTER(c_int)), ('ra', POINTER(c_double)), ('sa', POINTER(c_char_p)), ('derivs', POINTER(c_double)), ('hes', POINTER(c_double)), ('dig', POINTER(c_byte)), ('funcinfo', c_char_p), ('AE', c_void_p), ('f', c_void_p), ('tva', c_void_p), ('Errmsg', c_char_p), ('TMI', c_void_p), ('Private', c_char_p), ('nin', c_int), ('nout', c_int), ('nsin', c_int), ('nsout', c_int)]

    def __init__(self, args, fgh=0, fixed=None):
        super().__init__()
        self._encoded_strings = []
        self.n = len(args)
        self.at = (c_int * self.n)()
        _reals = []
        _strings = []
        nr = 0
        ns = 0
        for i, arg in enumerate(args):
            if arg.__class__ in native_numeric_types:
                _reals.append(arg)
                self.at[i] = nr
                nr += 1
                continue
            if isinstance(arg, str):
                arg = arg.encode('ascii')
                self._encoded_strings.append(arg)
            if isinstance(arg, bytes):
                _strings.append(arg)
                ns += 1
                self.at[i] = -ns
            else:
                raise RuntimeError(f'Unknown data type, {type(arg).__name__}, passed as argument {i} for an ASL ExternalFunction')
        self.nr = nr
        self.ra = (c_double * nr)(*_reals)
        self.sa = (c_char_p * ns)(*_strings)
        if fgh >= 1:
            self.derivs = (c_double * nr)(0.0)
        if fgh >= 2:
            self.hes = (c_double * ((nr + nr * nr) // 2))(0.0)
        if fixed:
            self.dig = (c_byte * nr)(0)
            for i, v in enumerate(fixed):
                if v:
                    r_idx = self.at[i]
                    if r_idx >= 0:
                        self.dig[r_idx] = 1