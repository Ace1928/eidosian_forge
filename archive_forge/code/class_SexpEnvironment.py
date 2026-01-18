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
class SexpEnvironment(Sexp):
    """Proxy for an R "environment" object.

    An R "environment" object can be thought of as a mix of a
    mapping (like a `dict`) and a scope. To make it more "Pythonic",
    both aspects are kept separate and the method `__getitem__` will
    get an item as it would for a Python `dict` while the method `find`
    will get an item as if it was a scope.

    As soon as R is initialized the following main environments become
    available to the user:
    - `globalenv`: The "workspace" for the current R process. This can
      be thought of as when `__name__ == '__main__'` in Python.
    - `baseenv`: The namespace of R's "base" package.
    """

    @_cdata_res_to_rinterface
    @_evaluated_promise
    def find(self, key: str, wantfun: bool=False) -> Sexp:
        """Find an item, starting with this R environment.

        Raises a `KeyError` if the key cannot be found.

        This method is called `find` because it is somewhat different
        from the method :meth:`get` in Python mappings such :class:`dict`.
        This is looking for a key across enclosing environments, returning
        the first key found."""
        if not isinstance(key, str):
            raise TypeError('The key must be a non-empty string.')
        elif not len(key):
            raise ValueError('The key must be a non-empty string.')
        with memorymanagement.rmemory() as rmemory:
            key_cchar = conversion._str_to_cchar(key, 'utf-8')
            symbol = rmemory.protect(openrlib.rlib.Rf_install(key_cchar))
            if wantfun:
                rho = self
                while rho.rid != emptyenv.rid:
                    res = rmemory.protect(_rinterface.findvar_in_frame_wrap(rho.__sexp__._cdata, symbol))
                    if _rinterface._TYPEOF(res) in (openrlib.rlib.CLOSXP, openrlib.rlib.BUILTINSXP):
                        break
                    res = openrlib.rlib.R_UnboundValue
                    rho = rho.enclos
            else:
                res = _rinterface._findvar(symbol, self.__sexp__._cdata)
        if res == openrlib.rlib.R_UnboundValue:
            raise KeyError("'%s' not found" % key)
        return res

    @_cdata_res_to_rinterface
    @_evaluated_promise
    def __getitem__(self, key: str) -> typing.Any:
        if not isinstance(key, str):
            raise TypeError('The key must be a non-empty string.')
        elif not len(key):
            raise ValueError('The key must be a non-empty string.')
        embedded.assert_isready()
        with memorymanagement.rmemory() as rmemory:
            key_cchar = conversion._str_to_cchar(key)
            symbol = rmemory.protect(openrlib.rlib.Rf_install(key_cchar))
            res = rmemory.protect(_rinterface.findvar_in_frame_wrap(self.__sexp__._cdata, symbol))
        if res == openrlib.rlib.R_UnboundValue:
            raise KeyError("'%s' not found" % key)
        return res

    def __setitem__(self, key: str, value) -> None:
        if not isinstance(key, str):
            raise TypeError('The key must be a non-empty string.')
        elif not len(key):
            raise ValueError('The key must be a non-empty string.')
        if self.__sexp__._cdata == openrlib.rlib.R_BaseEnv or self.__sexp__._cdata == openrlib.rlib.R_EmptyEnv:
            raise ValueError('Cannot remove variables from the base or empty environments.')
        with memorymanagement.rmemory() as rmemory:
            key_cchar = conversion._str_to_cchar(key)
            symbol = rmemory.protect(openrlib.rlib.Rf_install(key_cchar))
            cdata = rmemory.protect(conversion._get_cdata(value))
            cdata_copy = rmemory.protect(openrlib.rlib.Rf_duplicate(cdata))
            openrlib.rlib.Rf_defineVar(symbol, cdata_copy, self.__sexp__._cdata)

    def __len__(self) -> int:
        with memorymanagement.rmemory() as rmemory:
            symbols = rmemory.protect(openrlib.rlib.R_lsInternal(self.__sexp__._cdata, openrlib.rlib.TRUE))
            n = openrlib.rlib.Rf_xlength(symbols)
        return n

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError("'%s' not found" % key)
        if self.__sexp__ == baseenv.__sexp__:
            raise ValueError('Values from the R base environment cannot be removed.')
        if self.is_locked():
            ValueError('Cannot remove an item from a locked environment.')
        with memorymanagement.rmemory() as rmemory:
            key_cdata = rmemory.protect(openrlib.rlib.Rf_mkString(conversion._str_to_cchar(key)))
            _rinterface._remove(key_cdata, self.__sexp__._cdata, openrlib.rlib.Rf_ScalarLogical(openrlib.rlib.FALSE))

    @_cdata_res_to_rinterface
    def frame(self) -> 'typing.Union[NULLType, SexpEnvironment]':
        """Get the parent frame of the environment."""
        return openrlib.rlib.FRAME(self.__sexp__._cdata)

    @property
    @_cdata_res_to_rinterface
    def enclos(self) -> 'typing.Union[NULLType, SexpEnvironment]':
        """Get or set the enclosing environment."""
        return openrlib.rlib.ENCLOS(self.__sexp__._cdata)

    @enclos.setter
    def enclos(self, value: 'SexpEnvironment') -> None:
        assert isinstance(value, SexpEnvironment)
        openrlib.rlib.SET_ENCLOS(self.__sexp__._cdata, value.__sexp__._cdata)

    def keys(self) -> typing.Generator[str, None, None]:
        """Generator over the keys (symbols) in the environment."""
        with memorymanagement.rmemory() as rmemory:
            symbols = rmemory.protect(openrlib.rlib.R_lsInternal(self.__sexp__._cdata, openrlib.rlib.TRUE))
            n = openrlib.rlib.Rf_xlength(symbols)
            res = []
            for i in range(n):
                _ = _rinterface._string_getitem(symbols, i)
                if _ is None:
                    raise TypeError('R symbol string should not be able to be NA.')
                res.append(_)
        for e in res:
            yield e

    def __iter__(self) -> typing.Generator[str, None, None]:
        """See method `keys()`."""
        return self.keys()

    def is_locked(self) -> bool:
        return openrlib.rlib.R_EnvironmentIsLocked(self.__sexp__._cdata)