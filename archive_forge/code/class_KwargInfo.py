from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
class KwargInfo(T.Generic[_T]):
    """A description of a keyword argument to a meson function

    This is used to describe a value to the :func:typed_kwargs function.

    :param name: the name of the parameter
    :param types: A type or tuple of types that are allowed, or a :class:ContainerType
    :param required: Whether this is a required keyword argument. defaults to False
    :param listify: If true, then the argument will be listified before being
        checked. This is useful for cases where the Meson DSL allows a scalar or
        a container, but internally we only want to work with containers
    :param default: A default value to use if this isn't set. defaults to None,
        this may be safely set to a mutable type, as long as that type does not
        itself contain mutable types, typed_kwargs will copy the default
    :param since: Meson version in which this argument has been added. defaults to None
    :param since_message: An extra message to pass to FeatureNew when since is triggered
    :param deprecated: Meson version in which this argument has been deprecated. defaults to None
    :param deprecated_message: An extra message to pass to FeatureDeprecated
        when since is triggered
    :param validator: A callable that does additional validation. This is mainly
        intended for cases where a string is expected, but only a few specific
        values are accepted. Must return None if the input is valid, or a
        message if the input is invalid
    :param convertor: A callable that converts the raw input value into a
        different type. This is intended for cases such as the meson DSL using a
        string, but the implementation using an Enum. This should not do
        validation, just conversion.
    :param deprecated_values: a dictionary mapping a value to the version of
        meson it was deprecated in. The Value may be any valid value for this
        argument.
    :param since_values: a dictionary mapping a value to the version of meson it was
        added in.
    :param not_set_warning: A warning message that is logged if the kwarg is not
        set by the user.
    """

    def __init__(self, name: str, types: T.Union[T.Type[_T], T.Tuple[T.Union[T.Type[_T], ContainerTypeInfo], ...], ContainerTypeInfo], *, required: bool=False, listify: bool=False, default: T.Optional[_T]=None, since: T.Optional[str]=None, since_message: T.Optional[str]=None, since_values: T.Optional[T.Dict[T.Union[_T, ContainerTypeInfo, type], T.Union[str, T.Tuple[str, str]]]]=None, deprecated: T.Optional[str]=None, deprecated_message: T.Optional[str]=None, deprecated_values: T.Optional[T.Dict[T.Union[_T, ContainerTypeInfo, type], T.Union[str, T.Tuple[str, str]]]]=None, validator: T.Optional[T.Callable[[T.Any], T.Optional[str]]]=None, convertor: T.Optional[T.Callable[[_T], object]]=None, not_set_warning: T.Optional[str]=None):
        self.name = name
        self.types = types
        self.required = required
        self.listify = listify
        self.default = default
        self.since = since
        self.since_message = since_message
        self.since_values = since_values
        self.deprecated = deprecated
        self.deprecated_message = deprecated_message
        self.deprecated_values = deprecated_values
        self.validator = validator
        self.convertor = convertor
        self.not_set_warning = not_set_warning

    def evolve(self, *, name: T.Union[str, _NULL_T]=_NULL, required: T.Union[bool, _NULL_T]=_NULL, listify: T.Union[bool, _NULL_T]=_NULL, default: T.Union[_T, None, _NULL_T]=_NULL, since: T.Union[str, None, _NULL_T]=_NULL, since_message: T.Union[str, None, _NULL_T]=_NULL, since_values: T.Union[T.Dict[T.Union[_T, ContainerTypeInfo, type], T.Union[str, T.Tuple[str, str]]], None, _NULL_T]=_NULL, deprecated: T.Union[str, None, _NULL_T]=_NULL, deprecated_message: T.Union[str, None, _NULL_T]=_NULL, deprecated_values: T.Union[T.Dict[T.Union[_T, ContainerTypeInfo, type], T.Union[str, T.Tuple[str, str]]], None, _NULL_T]=_NULL, validator: T.Union[T.Callable[[_T], T.Optional[str]], None, _NULL_T]=_NULL, convertor: T.Union[T.Callable[[_T], TYPE_var], None, _NULL_T]=_NULL) -> 'KwargInfo':
        """Create a shallow copy of this KwargInfo, with modifications.

        This allows us to create a new copy of a KwargInfo with modifications.
        This allows us to use a shared kwarg that implements complex logic, but
        has slight differences in usage, such as being added to different
        functions in different versions of Meson.

        The use the _NULL special value here allows us to pass None, which has
        meaning in many of these cases. _NULL itself is never stored, always
        being replaced by either the copy in self, or the provided new version.
        """
        return type(self)(name if not isinstance(name, _NULL_T) else self.name, self.types, listify=listify if not isinstance(listify, _NULL_T) else self.listify, required=required if not isinstance(required, _NULL_T) else self.required, default=default if not isinstance(default, _NULL_T) else self.default, since=since if not isinstance(since, _NULL_T) else self.since, since_message=since_message if not isinstance(since_message, _NULL_T) else self.since_message, since_values=since_values if not isinstance(since_values, _NULL_T) else self.since_values, deprecated=deprecated if not isinstance(deprecated, _NULL_T) else self.deprecated, deprecated_message=deprecated_message if not isinstance(deprecated_message, _NULL_T) else self.deprecated_message, deprecated_values=deprecated_values if not isinstance(deprecated_values, _NULL_T) else self.deprecated_values, validator=validator if not isinstance(validator, _NULL_T) else self.validator, convertor=convertor if not isinstance(convertor, _NULL_T) else self.convertor)