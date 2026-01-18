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
class TraitType(BaseDescriptor, t.Generic[G, S]):
    """A base class for all trait types."""
    metadata: dict[str, t.Any] = {}
    allow_none: bool = False
    read_only: bool = False
    info_text: str = 'any value'
    default_value: t.Any = Undefined

    def __init__(self: TraitType[G, S], default_value: t.Any=Undefined, allow_none: bool=False, read_only: bool | None=None, help: str | None=None, config: t.Any=None, **kwargs: t.Any) -> None:
        """Declare a traitlet.

        If *allow_none* is True, None is a valid value in addition to any
        values that are normally valid. The default is up to the subclass.
        For most trait types, the default value for ``allow_none`` is False.

        If *read_only* is True, attempts to directly modify a trait attribute raises a TraitError.

        If *help* is a string, it documents the attribute's purpose.

        Extra metadata can be associated with the traitlet using the .tag() convenience method
        or by using the traitlet instance's .metadata dictionary.
        """
        if default_value is not Undefined:
            self.default_value = default_value
        if allow_none:
            self.allow_none = allow_none
        if read_only is not None:
            self.read_only = read_only
        self.help = help if help is not None else ''
        if self.help:
            self.__doc__ = self.help
        if len(kwargs) > 0:
            stacklevel = 1
            f = inspect.currentframe()
            assert f is not None
            while f.f_code.co_name == '__init__':
                stacklevel += 1
                f = f.f_back
                assert f is not None
            mod = f.f_globals.get('__name__') or ''
            pkg = mod.split('.', 1)[0]
            key = ('metadata-tag', pkg, *sorted(kwargs))
            if should_warn(key):
                warn(f"metadata {kwargs} was set from the constructor. With traitlets 4.1, metadata should be set using the .tag() method, e.g., Int().tag(key1='value1', key2='value2')", DeprecationWarning, stacklevel=stacklevel)
            if len(self.metadata) > 0:
                self.metadata = self.metadata.copy()
                self.metadata.update(kwargs)
            else:
                self.metadata = kwargs
        else:
            self.metadata = self.metadata.copy()
        if config is not None:
            self.metadata['config'] = config
        if help is not None:
            self.metadata['help'] = help

    def from_string(self, s: str) -> G | None:
        """Get a value from a config string

        such as an environment variable or CLI arguments.

        Traits can override this method to define their own
        parsing of config strings.

        .. seealso:: item_from_string

        .. versionadded:: 5.0
        """
        if self.allow_none and s == 'None':
            return None
        return s

    def default(self, obj: t.Any=None) -> G | None:
        """The default generator for this trait

        Notes
        -----
        This method is registered to HasTraits classes during ``class_init``
        in the same way that dynamic defaults defined by ``@default`` are.
        """
        if self.default_value is not Undefined:
            return t.cast(G, self.default_value)
        elif hasattr(self, 'make_dynamic_default'):
            return t.cast(G, self.make_dynamic_default())
        else:
            return t.cast(G, self.default_value)

    def get_default_value(self) -> G | None:
        """DEPRECATED: Retrieve the static default value for this trait.
        Use self.default_value instead
        """
        warn('get_default_value is deprecated in traitlets 4.0: use the .default_value attribute', DeprecationWarning, stacklevel=2)
        return t.cast(G, self.default_value)

    def init_default_value(self, obj: t.Any) -> G | None:
        """DEPRECATED: Set the static default value for the trait type."""
        warn('init_default_value is deprecated in traitlets 4.0, and may be removed in the future', DeprecationWarning, stacklevel=2)
        value = self._validate(obj, self.default_value)
        obj._trait_values[self.name] = value
        return value

    def get(self, obj: HasTraits, cls: type[t.Any] | None=None) -> G | None:
        assert self.name is not None
        try:
            value = obj._trait_values[self.name]
        except KeyError:
            default = obj.trait_defaults(self.name)
            if default is Undefined:
                warn('Explicit using of Undefined as the default value is deprecated in traitlets 5.0, and may cause exceptions in the future.', DeprecationWarning, stacklevel=2)
            _cross_validation_lock = obj._cross_validation_lock
            try:
                obj._cross_validation_lock = True
                value = self._validate(obj, default)
            finally:
                obj._cross_validation_lock = _cross_validation_lock
            obj._trait_values[self.name] = value
            obj._notify_observers(Bunch(name=self.name, value=value, owner=obj, type='default'))
            return t.cast(G, value)
        except Exception as e:
            raise TraitError('Unexpected error in TraitType: default value not set properly') from e
        else:
            return t.cast(G, value)

    @t.overload
    def __get__(self, obj: None, cls: type[t.Any]) -> Self:
        ...

    @t.overload
    def __get__(self, obj: t.Any, cls: type[t.Any]) -> G:
        ...

    def __get__(self, obj: HasTraits | None, cls: type[t.Any]) -> Self | G:
        """Get the value of the trait by self.name for the instance.

        Default values are instantiated when :meth:`HasTraits.__new__`
        is called.  Thus by the time this method gets called either the
        default value or a user defined value (they called :meth:`__set__`)
        is in the :class:`HasTraits` instance.
        """
        if obj is None:
            return self
        else:
            return t.cast(G, self.get(obj, cls))

    def set(self, obj: HasTraits, value: S) -> None:
        new_value = self._validate(obj, value)
        assert self.name is not None
        try:
            old_value = obj._trait_values[self.name]
        except KeyError:
            old_value = self.default_value
        obj._trait_values[self.name] = new_value
        try:
            silent = bool(old_value == new_value)
        except Exception:
            silent = False
        if silent is not True:
            obj._notify_trait(self.name, old_value, new_value)

    def __set__(self, obj: HasTraits, value: S) -> None:
        """Set the value of the trait by self.name for the instance.

        Values pass through a validation stage where errors are raised when
        impropper types, or types that cannot be coerced, are encountered.
        """
        if self.read_only:
            raise TraitError('The "%s" trait is read-only.' % self.name)
        self.set(obj, value)

    def _validate(self, obj: t.Any, value: t.Any) -> G | None:
        if value is None and self.allow_none:
            return value
        if hasattr(self, 'validate'):
            value = self.validate(obj, value)
        if obj._cross_validation_lock is False:
            value = self._cross_validate(obj, value)
        return t.cast(G, value)

    def _cross_validate(self, obj: t.Any, value: t.Any) -> G | None:
        if self.name in obj._trait_validators:
            proposal = Bunch({'trait': self, 'value': value, 'owner': obj})
            value = obj._trait_validators[self.name](obj, proposal)
        elif hasattr(obj, '_%s_validate' % self.name):
            meth_name = '_%s_validate' % self.name
            cross_validate = getattr(obj, meth_name)
            deprecated_method(cross_validate, obj.__class__, meth_name, 'use @validate decorator instead.')
            value = cross_validate(value, self)
        return t.cast(G, value)

    def __or__(self, other: TraitType[t.Any, t.Any]) -> Union:
        if isinstance(other, Union):
            return Union([self, *other.trait_types])
        else:
            return Union([self, other])

    def info(self) -> str:
        return self.info_text

    def error(self, obj: HasTraits | None, value: t.Any, error: Exception | None=None, info: str | None=None) -> t.NoReturn:
        """Raise a TraitError

        Parameters
        ----------
        obj : HasTraits or None
            The instance which owns the trait. If not
            object is given, then an object agnostic
            error will be raised.
        value : any
            The value that caused the error.
        error : Exception (default: None)
            An error that was raised by a child trait.
            The arguments of this exception should be
            of the form ``(value, info, *traits)``.
            Where the ``value`` and ``info`` are the
            problem value, and string describing the
            expected value. The ``traits`` are a series
            of :class:`TraitType` instances that are
            "children" of this one (the first being
            the deepest).
        info : str (default: None)
            A description of the expected value. By
            default this is inferred from this trait's
            ``info`` method.
        """
        if error is not None:
            error.args += (self,)
            if self.name is not None:
                chain = ' of '.join((describe('a', t) for t in error.args[2:]))
                if obj is not None:
                    error.args = ("The '{}' trait of {} instance contains {} which expected {}, not {}.".format(self.name, describe('an', obj), chain, error.args[1], describe('the', error.args[0])),)
                else:
                    error.args = ("The '{}' trait contains {} which expected {}, not {}.".format(self.name, chain, error.args[1], describe('the', error.args[0])),)
            raise error
        if self.name is None:
            raise TraitError(value, info or self.info(), self)
        if obj is not None:
            e = "The '{}' trait of {} instance expected {}, not {}.".format(self.name, class_of(obj), info or self.info(), describe('the', value))
        else:
            e = "The '{}' trait expected {}, not {}.".format(self.name, info or self.info(), describe('the', value))
        raise TraitError(e)

    def get_metadata(self, key: str, default: t.Any=None) -> t.Any:
        """DEPRECATED: Get a metadata value.

        Use .metadata[key] or .metadata.get(key, default) instead.
        """
        if key == 'help':
            msg = 'use the instance .help string directly, like x.help'
        else:
            msg = 'use the instance .metadata dictionary directly, like x.metadata[key] or x.metadata.get(key, default)'
        warn('Deprecated in traitlets 4.1, ' + msg, DeprecationWarning, stacklevel=2)
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: t.Any) -> None:
        """DEPRECATED: Set a metadata key/value.

        Use .metadata[key] = value instead.
        """
        if key == 'help':
            msg = 'use the instance .help string directly, like x.help = value'
        else:
            msg = 'use the instance .metadata dictionary directly, like x.metadata[key] = value'
        warn('Deprecated in traitlets 4.1, ' + msg, DeprecationWarning, stacklevel=2)
        self.metadata[key] = value

    def tag(self, **metadata: t.Any) -> Self:
        """Sets metadata and returns self.

        This allows convenient metadata tagging when initializing the trait, such as:

        Examples
        --------
        >>> Int(0).tag(config=True, sync=True)
        <traitlets.traitlets.Int object at ...>

        """
        maybe_constructor_keywords = set(metadata.keys()).intersection({'help', 'allow_none', 'read_only', 'default_value'})
        if maybe_constructor_keywords:
            warn('The following attributes are set in using `tag`, but seem to be constructor keywords arguments: %s ' % maybe_constructor_keywords, UserWarning, stacklevel=2)
        self.metadata.update(metadata)
        return self

    def default_value_repr(self) -> str:
        return repr(self.default_value)