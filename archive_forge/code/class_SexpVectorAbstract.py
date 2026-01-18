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
class SexpVectorAbstract(SupportsSEXP, typing.Generic[VT], metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def _R_TYPE(self):
        pass

    @property
    @abc.abstractmethod
    def _R_SIZEOF_ELT(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _CAST_IN(o):
        pass

    @staticmethod
    @abc.abstractmethod
    def _R_SET_VECTOR_ELT(x, i, v):
        pass

    @staticmethod
    @abc.abstractmethod
    def _R_VECTOR_ELT(x, i):
        pass

    @staticmethod
    @abc.abstractmethod
    def _R_GET_PTR(o):
        pass

    @classmethod
    @_cdata_res_to_rinterface
    def from_iterable(cls, iterable, populate_func=None, set_elt=None, cast_value=None) -> VT:
        """Create an R vector/array from an iterable."""
        if not embedded.isready():
            raise embedded.RNotReadyError('Embedded R is not ready to use.')
        if populate_func is None:
            populate_func = _populate_r_vector
        if set_elt is None:
            set_elt = cls._R_SET_VECTOR_ELT
        if cast_value is None:
            cast_value = cls._CAST_IN
        n = len(iterable)
        with memorymanagement.rmemory() as rmemory:
            r_vector = rmemory.protect(openrlib.rlib.Rf_allocVector(cls._R_TYPE, n))
            populate_func(iterable, r_vector, set_elt, cast_value)
        return r_vector

    @classmethod
    def _raise_incompatible_C_size(cls, mview):
        msg = 'Incompatible C type sizes. The R array type is "{r_type}" with {r_size} byte{r_size_pl} per item while the Python array type is "{py_type}" with {py_size} byte{py_size_pl} per item.'.format(r_type=cls._R_TYPE, r_size=cls._R_SIZEOF_ELT, r_size_pl='s' if cls._R_SIZEOF_ELT > 1 else '', py_type=mview.format, py_size=mview.itemsize, py_size_pl='s' if mview.itemsize > 1 else '')
        raise ValueError(msg)

    @classmethod
    def _check_C_compatible(cls, mview):
        return mview.itemsize == cls._R_SIZEOF_ELT

    @classmethod
    @_cdata_res_to_rinterface
    def from_memoryview(cls, mview: memoryview) -> VT:
        """Create an R vector/array from a memoryview.

        The memoryview must be contiguous, and the C representation
        for the vector must be compatible between R and Python. If
        not the case, a :class:`ValueError` exception with will be
        raised."""
        if not embedded.isready():
            raise embedded.RNotReadyError('Embedded R is not ready to use.')
        if not mview.contiguous:
            raise ValueError('The memory view must be contiguous.')
        if not cls._check_C_compatible(mview):
            cls._raise_incompatible_C_size(mview)
        r_vector = None
        n = len(mview)
        with memorymanagement.rmemory() as rmemory:
            r_vector = rmemory.protect(openrlib.rlib.Rf_allocVector(cls._R_TYPE, n))
            dest_ptr = cls._R_GET_PTR(r_vector)
            src_ptr = _rinterface.ffi.from_buffer(mview)
            nbytes = n * mview.itemsize
            _rinterface.ffi.memmove(dest_ptr, src_ptr, nbytes)
        return r_vector

    @classmethod
    def from_object(cls, obj) -> VT:
        """Create an R vector/array from a Python object, if possible.

        An exception :class:`ValueError` will be raised if not possible."""
        try:
            mv = memoryview(obj)
            res = cls.from_memoryview(mv)
        except (TypeError, ValueError):
            try:
                res = cls.from_iterable(obj)
            except ValueError:
                msg = 'The class methods from_memoryview() and from_iterable() both failed to make a {} from an object of class {}'.format(cls, type(obj))
                raise ValueError(msg)
        return res

    def __getitem__(self, i: typing.Union[int, slice]) -> typing.Union[Sexp, VT, typing.Any]:
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            res = conversion._cdata_to_rinterface(self._R_VECTOR_ELT(cdata, i_c))
        elif isinstance(i, slice):
            res = self.from_iterable([self._R_VECTOR_ELT(cdata, i_c) for i_c in range(*i.indices(len(self)))], cast_value=lambda x: x)
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))
        return res

    def __setitem__(self, i: typing.Union[int, slice], value) -> None:
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            if isinstance(value, Sexp):
                val_cdata = value.__sexp__._cdata
            else:
                val_cdata = conversion._python_to_cdata(value)
            self._R_SET_VECTOR_ELT(cdata, i_c, val_cdata)
        elif isinstance(i, slice):
            for i_c, v in zip(range(*i.indices(len(self))), value):
                self._R_SET_VECTOR_ELT(cdata, i_c, v.__sexp__._cdata)
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))

    def __len__(self) -> int:
        return openrlib.rlib.Rf_xlength(self.__sexp__._cdata)

    def __iter__(self) -> typing.Iterator[typing.Union[Sexp, VT, typing.Any]]:
        for i in range(len(self)):
            yield self[i]

    def index(self, item: typing.Any) -> int:
        for i, e in enumerate(self):
            if e == item:
                return i
        raise ValueError("'%s' is not in R vector" % item)