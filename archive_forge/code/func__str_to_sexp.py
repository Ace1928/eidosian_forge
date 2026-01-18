from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _str_to_sexp(val: str):
    rlib = openrlib.rlib
    s = rlib.Rf_protect(rlib.Rf_allocVector(rlib.STRSXP, 1))
    charval = _str_to_charsxp(val)
    rlib.SET_STRING_ELT(s, 0, charval)
    rlib.Rf_unprotect(1)
    return s