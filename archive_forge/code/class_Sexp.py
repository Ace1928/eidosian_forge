import abc
import collections.abc
from collections import OrderedDict
import enum
import itertools
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
from rpy2.rinterface_lib._rinterface_capi import _evaluated_promise
from rpy2.rinterface_lib._rinterface_capi import SupportsSEXP
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
from rpy2.rinterface_lib import na_values
class Sexp(SupportsSEXP):
    """Base class for R objects.

    The name of a class corresponds to the name SEXP
    used in R's C API."""
    __slots__ = ('_sexpobject',)

    def __init__(self, sexp: typing.Union[SupportsSEXP, '_rinterface.SexpCapsule', '_rinterface.UninitializedRCapsule']):
        if isinstance(sexp, SupportsSEXP):
            self._sexpobject = sexp.__sexp__
        elif isinstance(sexp, _rinterface.CapsuleBase):
            self._sexpobject = sexp
        else:
            raise ValueError('The constructor must be called with an instance of rpy2.rinterface.Sexp or an instance of rpy2.rinterface._rinterface.SexpCapsule')

    def __repr__(self) -> str:
        return super().__repr__() + ' [%s]' % self.typeof

    @property
    def __sexp__(self) -> typing.Union['_rinterface.SexpCapsule', '_rinterface.UninitializedRCapsule']:
        """Access to the underlying C pointer to the R object.

        When assigning a new SexpCapsule to this attribute, the
        R C-level type of the new capsule must be equal to the
        type of the old capsule. A ValueError is raised otherwise."""
        return self._sexpobject

    @__sexp__.setter
    def __sexp__(self, value: typing.Union['_rinterface.SexpCapsule', '_rinterface.UninitializedRCapsule']) -> None:
        assert isinstance(value, _rinterface.SexpCapsule)
        if value.typeof != self.__sexp__.typeof:
            raise ValueError('New capsule type mismatch: %s' % RTYPES(value.typeof))
        self._sexpobject = value

    @property
    def __sexp_refcount__(self) -> int:
        """Count the number of independent Python references to
        the underlying R object."""
        return _rinterface._R_PRESERVED[_rinterface.get_rid(self.__sexp__._cdata)]

    def __getstate__(self) -> bytes:
        with memorymanagement.rmemory() as rmemory:
            ser = rmemory.protect(_rinterface.serialize(self.__sexp__._cdata, globalenv.__sexp__._cdata))
            n = openrlib.rlib.Rf_xlength(ser)
            res = bytes(_rinterface.ffi.buffer(openrlib.rlib.RAW(ser), n))
        return res

    def __setstate__(self, state: bytes) -> None:
        self._sexpobject = unserialize(state)

    @property
    def rclass(self) -> 'StrSexpVector':
        """Get or set the R "class" attribute for the object."""
        return rclass_get(self.__sexp__)

    @rclass.setter
    def rclass(self, value: 'typing.Union[StrSexpVector, str]'):
        rclass_set(self.__sexp__, value)

    @property
    def rid(self) -> int:
        """ID of the underlying R object (memory address)."""
        return _rinterface.get_rid(self.__sexp__._cdata)

    @property
    def typeof(self) -> RTYPES:
        return RTYPES(_rinterface._TYPEOF(self.__sexp__._cdata))

    @property
    def named(self) -> int:
        return _rinterface._NAMED(self.__sexp__._cdata)

    @conversion._cdata_res_to_rinterface
    def list_attrs(self) -> 'typing.Union[StrSexpVector, str]':
        return _rinterface._list_attrs(self.__sexp__._cdata)

    @conversion._cdata_res_to_rinterface
    def do_slot(self, name: str) -> None:
        _rinterface._assert_valid_slotname(name)
        cchar = conversion._str_to_cchar(name)
        with memorymanagement.rmemory() as rmemory:
            name_r = rmemory.protect(openrlib.rlib.Rf_install(cchar))
            if not _rinterface._has_slot(self.__sexp__._cdata, name_r):
                raise LookupError(name)
            res = openrlib.rlib.R_do_slot(self.__sexp__._cdata, name_r)
        return res

    def do_slot_assign(self, name: str, value) -> None:
        _rinterface._assert_valid_slotname(name)
        cchar = conversion._str_to_cchar(name)
        with memorymanagement.rmemory() as rmemory:
            name_r = rmemory.protect(openrlib.rlib.Rf_install(cchar))
            cdata = rmemory.protect(conversion._get_cdata(value))
            openrlib.rlib.R_do_slot_assign(self.__sexp__._cdata, name_r, cdata)

    @conversion._cdata_res_to_rinterface
    def get_attrib(self, name: str) -> 'Sexp':
        res = openrlib.rlib.Rf_getAttrib(self.__sexp__._cdata, conversion._str_to_charsxp(name))
        return res

    def rsame(self, sexp) -> bool:
        if isinstance(sexp, Sexp):
            return self.__sexp__._cdata == sexp.__sexp__._cdata
        elif isinstance(sexp, _rinterface.SexpCapsule):
            return sexp._cdata == sexp._cdata
        else:
            raise ValueError('Not an R object.')

    @property
    def names(self) -> 'Sexp':
        return baseenv['names'](self)

    @names.setter
    def names(self, value) -> None:
        if not isinstance(value, StrSexpVector):
            raise ValueError('The new names should be a StrSexpVector.')
        openrlib.rlib.Rf_namesgets(self.__sexp__._cdata, value.__sexp__._cdata)

    @property
    @conversion._cdata_res_to_rinterface
    def names_from_c_attribute(self) -> 'Sexp':
        return openrlib.rlib.Rf_getAttrib(self.__sexp__._cdata, openrlib.rlib.R_NameSymbol)