import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
@ffi_proxy.callback(ffi_proxy._evaluate_in_r_def, openrlib._rinterface_cffi)
def _evaluate_in_r(rargs: FFI.CData) -> FFI.CData:
    rlib = openrlib.rlib
    try:
        rargs = rlib.CDR(rargs)
        cdata = rlib.CAR(rargs)
        if _TYPEOF(cdata) != rlib.EXTPTRSXP:
            logger.error('The fist item is not an R external pointer.')
            return rlib.R_NilValue
        handle = rlib.R_ExternalPtrAddr(cdata)
        func = ffi.from_handle(handle)
        pyargs = []
        pykwargs = {}
        pyfunc_params_iter = inspect.signature(func).parameters.items()
        py_posonly = []
        py_has_ellipsis = False
        py_positionalorkw = []
        for paramname, paramval in pyfunc_params_iter:
            if paramval.kind is inspect.Parameter.POSITIONAL_ONLY:
                py_posonly.append(paramname)
            elif paramval.kind is inspect.Parameter.VAR_POSITIONAL or paramval.kind is inspect.Parameter.VAR_KEYWORD:
                py_has_ellipsis = True
            elif paramval.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                py_positionalorkw.append(paramname)
            else:
                break
        rarg_i = -1
        rargs = rlib.CDR(rargs)
        while rargs != rlib.R_NilValue:
            rarg_i += 1
            cdata = rlib.CAR(rargs)
            if rlib.Rf_isNull(rlib.TAG(rargs)):
                pyargs.append(conversion._cdata_to_rinterface(cdata))
            else:
                rname = rlib.PRINTNAME(rlib.TAG(rargs))
                name = conversion._cchar_to_str(rlib.R_CHAR(rname), conversion._R_ENC_PY[openrlib.rlib.Rf_getCharCE(rname)])
                if rarg_i < len(py_posonly):
                    if py_posonly[rarg_i] == name:
                        pyargs.append(conversion._cdata_to_rinterface(cdata))
                    else:
                        raise RuntimeError(f'Parameter name mismatch. R call considering the argument "{py_posonly[rarg_i]}" as a position-independent keyword argument while it is positional-only in the rternalized Python function.')
                elif py_has_ellipsis and rarg_i < len(py_posonly) + len(py_positionalorkw):
                    if py_positionalorkw[rarg_i - len(py_posonly)] == name:
                        pyargs.append(conversion._cdata_to_rinterface(cdata))
                    else:
                        raise RuntimeError(f'Parameter name mismatch. R call considering the argument "{py_posonly[rarg_i]}" as a position-independent keyword argument while it is positional-or-keyword followed by an positional ellipsis `*args` in the rternalized Python function.')
                else:
                    pykwargs[name] = conversion._cdata_to_rinterface(cdata)
            rargs = rlib.CDR(rargs)
        res = func(*pyargs, **pykwargs)
        if hasattr(res, '_sexpobject') and isinstance(res._sexpobject, SexpCapsule):
            return res._sexpobject._cdata
        else:
            return conversion._python_to_cdata(res)
    except Exception as e:
        logger.error('%s: rternalized %s' % (type(e).__name__, e))
        return rlib.R_NilValue