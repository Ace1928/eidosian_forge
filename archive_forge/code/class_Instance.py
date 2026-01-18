from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
class Instance(ClassBasedTraitType[T, T]):
    """A trait whose value must be an instance of a specified class.

    The value can also be an instance of a subclass of the specified class.

    Subclasses can declare default classes by overriding the klass attribute
    """
    klass: str | type[T] | None = None
    if t.TYPE_CHECKING:

        @t.overload
        def __init__(self: Instance[T], klass: type[T]=..., args: tuple[t.Any, ...] | None=..., kw: dict[str, t.Any] | None=..., allow_none: Literal[False]=..., read_only: bool | None=..., help: str | None=..., **kwargs: t.Any) -> None:
            ...

        @t.overload
        def __init__(self: Instance[T | None], klass: type[T]=..., args: tuple[t.Any, ...] | None=..., kw: dict[str, t.Any] | None=..., allow_none: Literal[True]=..., read_only: bool | None=..., help: str | None=..., **kwargs: t.Any) -> None:
            ...

        @t.overload
        def __init__(self: Instance[t.Any], klass: str | None=..., args: tuple[t.Any, ...] | None=..., kw: dict[str, t.Any] | None=..., allow_none: Literal[False]=..., read_only: bool | None=..., help: str | None=..., **kwargs: t.Any) -> None:
            ...

        @t.overload
        def __init__(self: Instance[t.Any | None], klass: str | None=..., args: tuple[t.Any, ...] | None=..., kw: dict[str, t.Any] | None=..., allow_none: Literal[True]=..., read_only: bool | None=..., help: str | None=..., **kwargs: t.Any) -> None:
            ...

    def __init__(self, klass: str | type[T] | None=None, args: tuple[t.Any, ...] | None=None, kw: dict[str, t.Any] | None=None, allow_none: bool=False, read_only: bool | None=None, help: str | None=None, **kwargs: t.Any) -> None:
        """Construct an Instance trait.

        This trait allows values that are instances of a particular
        class or its subclasses.  Our implementation is quite different
        from that of enthough.traits as we don't allow instances to be used
        for klass and we handle the ``args`` and ``kw`` arguments differently.

        Parameters
        ----------
        klass : class, str
            The class that forms the basis for the trait.  Class names
            can also be specified as strings, like 'foo.bar.Bar'.
        args : tuple
            Positional arguments for generating the default value.
        kw : dict
            Keyword arguments for generating the default value.
        allow_none : bool [ default False ]
            Indicates whether None is allowed as a value.
        **kwargs
            Extra kwargs passed to `ClassBasedTraitType`

        Notes
        -----
        If both ``args`` and ``kw`` are None, then the default value is None.
        If ``args`` is a tuple and ``kw`` is a dict, then the default is
        created as ``klass(*args, **kw)``.  If exactly one of ``args`` or ``kw`` is
        None, the None is replaced by ``()`` or ``{}``, respectively.
        """
        if klass is None:
            klass = self.klass
        if klass is not None and (inspect.isclass(klass) or isinstance(klass, str)):
            self.klass = klass
        else:
            raise TraitError('The klass attribute must be a class not: %r' % klass)
        if kw is not None and (not isinstance(kw, dict)):
            raise TraitError("The 'kw' argument must be a dict or None.")
        if args is not None and (not isinstance(args, tuple)):
            raise TraitError("The 'args' argument must be a tuple or None.")
        self.default_args = args
        self.default_kwargs = kw
        super().__init__(allow_none=allow_none, read_only=read_only, help=help, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> T | None:
        assert self.klass is not None
        if self.allow_none and value is None:
            return value
        if isinstance(value, self.klass):
            return t.cast(T, value)
        else:
            self.error(obj, value)

    def info(self) -> str:
        if isinstance(self.klass, str):
            result = add_article(self.klass)
        else:
            result = describe('a', self.klass)
        if self.allow_none:
            result += ' or None'
        return result

    def instance_init(self, obj: t.Any) -> None:
        self._resolve_classes()

    def _resolve_classes(self) -> None:
        if isinstance(self.klass, str):
            self.klass = self._resolve_string(self.klass)

    def make_dynamic_default(self) -> T | None:
        if self.default_args is None and self.default_kwargs is None:
            return None
        assert self.klass is not None
        return self.klass(*(self.default_args or ()), **self.default_kwargs or {})

    def default_value_repr(self) -> str:
        return repr(self.make_dynamic_default())

    def from_string(self, s: str) -> T | None:
        return t.cast(T, _safe_literal_eval(s))