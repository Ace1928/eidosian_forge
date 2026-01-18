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
def _list_attrs(cdata: FFI.CData) -> FFI.CData:
    rlib = openrlib.rlib
    attrs = rlib.ATTRIB(cdata)
    nvalues = rlib.Rf_length(attrs)
    if rlib.Rf_isList(cdata):
        namesattr = rlib.Rf_getAttrib(cdata, rlib.R_NamesSymbol)
        if namesattr != rlib.R_NilValue:
            nvalues += 1
    else:
        namesattr = rlib.R_NilValue
    with memorymanagement.rmemory() as rmemory:
        names = rmemory.protect(rlib.Rf_allocVector(rlib.STRSXP, nvalues))
        attr_i = 0
        if namesattr != rlib.R_NilValue:
            rlib.SET_STRING_ELT(names, attr_i, rlib.PRINTNAME(rlib.R_NamesSymbol))
            attr_i += 1
        while attrs != rlib.R_NilValue:
            tag = rlib.TAG(attrs)
            if _TYPEOF(tag) == rlib.SYMSXP:
                rlib.SET_STRING_ELT(names, attr_i, rlib.PRINTNAME(tag))
            else:
                rlib.SET_STRING_ELT(names, attr_i, rlib.R_BlankString)
            attrs = rlib.CDR(attrs)
            attr_i += 1
    return names