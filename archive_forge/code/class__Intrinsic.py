import os
import uuid
import weakref
import collections
import functools
import numba
from numba.core import types, errors, utils, config
from numba.core.typing.typeof import typeof_impl  # noqa: F401
from numba.core.typing.asnumbatype import as_numba_type  # noqa: F401
from numba.core.typing.templates import infer, infer_getattr  # noqa: F401
from numba.core.imputils import (  # noqa: F401
from numba.core.datamodel import models   # noqa: F401
from numba.core.datamodel import register_default as register_model  # noqa: F401, E501
from numba.core.pythonapi import box, unbox, reflect, NativeValue  # noqa: F401
from numba._helperlib import _import_cython_function  # noqa: F401
from numba.core.serialize import ReduceMixin
class _Intrinsic(ReduceMixin):
    """
    Dummy callable for intrinsic
    """
    _memo = weakref.WeakValueDictionary()
    _recent = collections.deque(maxlen=config.FUNCTION_CACHE_SIZE)
    __uuid = None

    def __init__(self, name, defn, prefer_literal=False, **kwargs):
        self._ctor_kwargs = kwargs
        self._name = name
        self._defn = defn
        self._prefer_literal = prefer_literal
        functools.update_wrapper(self, defn)

    @property
    def _uuid(self):
        """
        An instance-specific UUID, to avoid multiple deserializations of
        a given instance.

        Note this is lazily-generated, for performance reasons.
        """
        u = self.__uuid
        if u is None:
            u = str(uuid.uuid1())
            self._set_uuid(u)
        return u

    def _set_uuid(self, u):
        assert self.__uuid is None
        self.__uuid = u
        self._memo[u] = self
        self._recent.append(self)

    def _register(self):
        from numba.core.typing.templates import make_intrinsic_template, infer_global
        template = make_intrinsic_template(self, self._defn, self._name, prefer_literal=self._prefer_literal, kwargs=self._ctor_kwargs)
        infer(template)
        infer_global(self, types.Function(template))

    def __call__(self, *args, **kwargs):
        """
        This is only defined to pretend to be a callable from CPython.
        """
        msg = '{0} is not usable in pure-python'.format(self)
        raise NotImplementedError(msg)

    def __repr__(self):
        return '<intrinsic {0}>'.format(self._name)

    def __deepcopy__(self, memo):
        return self

    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
        return dict(uuid=self._uuid, name=self._name, defn=self._defn)

    @classmethod
    def _rebuild(cls, uuid, name, defn):
        """
        NOTE: part of ReduceMixin protocol
        """
        try:
            return cls._memo[uuid]
        except KeyError:
            llc = cls(name=name, defn=defn)
            llc._register()
            llc._set_uuid(uuid)
            return llc