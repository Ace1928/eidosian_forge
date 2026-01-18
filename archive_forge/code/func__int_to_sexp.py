from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _int_to_sexp(val: int):
    rlib = openrlib.rlib
    if val > _rinterface._MAX_INT:
        raise ValueError(f"The Python integer {val} is larger than {_rinterface._MAX_INT} (R's largest possible integer).")
    s = rlib.Rf_protect(rlib.Rf_allocVector(rlib.INTSXP, 1))
    openrlib.SET_INTEGER_ELT(s, 0, val)
    rlib.Rf_unprotect(1)
    return s