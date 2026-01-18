from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
class Generic:
    """Abstract base class for generic types.

    A generic type is typically declared by inheriting from
    this class parameterized with one or more type variables.
    For example, a generic mapping type might be defined as::

      class Mapping(Generic[KT, VT]):
          def __getitem__(self, key: KT) -> VT:
              ...
          # Etc.

    This class can then be used as follows::

      def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
          try:
              return mapping[key]
          except KeyError:
              return default
    """
    __slots__ = ()
    _is_protocol = False

    @_tp_cache
    def __class_getitem__(cls, params):
        """Parameterizes a generic class.

        At least, parameterizing a generic class is the *main* thing this method
        does. For example, for some generic class `Foo`, this is called when we
        do `Foo[int]` - there, with `cls=Foo` and `params=int`.

        However, note that this method is also called when defining generic
        classes in the first place with `class Foo(Generic[T]): ...`.
        """
        if not isinstance(params, tuple):
            params = (params,)
        params = tuple((_type_convert(p) for p in params))
        if cls in (Generic, Protocol):
            if not params:
                raise TypeError(f'Parameter list to {cls.__qualname__}[...] cannot be empty')
            if not all((_is_typevar_like(p) for p in params)):
                raise TypeError(f'Parameters to {cls.__name__}[...] must all be type variables or parameter specification variables.')
            if len(set(params)) != len(params):
                raise TypeError(f'Parameters to {cls.__name__}[...] must all be unique')
        else:
            for param in cls.__parameters__:
                prepare = getattr(param, '__typing_prepare_subst__', None)
                if prepare is not None:
                    params = prepare(cls, params)
            _check_generic(cls, params, len(cls.__parameters__))
            new_args = []
            for param, new_arg in zip(cls.__parameters__, params):
                if isinstance(param, TypeVarTuple):
                    new_args.extend(new_arg)
                else:
                    new_args.append(new_arg)
            params = tuple(new_args)
        return _GenericAlias(cls, params, _paramspec_tvars=True)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        tvars = []
        if '__orig_bases__' in cls.__dict__:
            error = Generic in cls.__orig_bases__
        else:
            error = Generic in cls.__bases__ and cls.__name__ != 'Protocol' and (type(cls) != _TypedDictMeta)
        if error:
            raise TypeError('Cannot inherit from plain Generic')
        if '__orig_bases__' in cls.__dict__:
            tvars = _collect_parameters(cls.__orig_bases__)
            gvars = None
            for base in cls.__orig_bases__:
                if isinstance(base, _GenericAlias) and base.__origin__ is Generic:
                    if gvars is not None:
                        raise TypeError('Cannot inherit from Generic[...] multiple times.')
                    gvars = base.__parameters__
            if gvars is not None:
                tvarset = set(tvars)
                gvarset = set(gvars)
                if not tvarset <= gvarset:
                    s_vars = ', '.join((str(t) for t in tvars if t not in gvarset))
                    s_args = ', '.join((str(g) for g in gvars))
                    raise TypeError(f'Some type variables ({s_vars}) are not listed in Generic[{s_args}]')
                tvars = gvars
        cls.__parameters__ = tuple(tvars)