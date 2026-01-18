from __future__ import annotations as _annotations
import dataclasses
import inspect
import typing
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Any, ClassVar
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, Unpack
from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .warnings import PydanticDeprecatedSince20
def Field(default: Any=PydanticUndefined, *, default_factory: typing.Callable[[], Any] | None=_Unset, alias: str | None=_Unset, alias_priority: int | None=_Unset, validation_alias: str | AliasPath | AliasChoices | None=_Unset, serialization_alias: str | None=_Unset, title: str | None=_Unset, description: str | None=_Unset, examples: list[Any] | None=_Unset, exclude: bool | None=_Unset, discriminator: str | types.Discriminator | None=_Unset, json_schema_extra: JsonDict | typing.Callable[[JsonDict], None] | None=_Unset, frozen: bool | None=_Unset, validate_default: bool | None=_Unset, repr: bool=_Unset, init: bool | None=_Unset, init_var: bool | None=_Unset, kw_only: bool | None=_Unset, pattern: str | None=_Unset, strict: bool | None=_Unset, gt: float | None=_Unset, ge: float | None=_Unset, lt: float | None=_Unset, le: float | None=_Unset, multiple_of: float | None=_Unset, allow_inf_nan: bool | None=_Unset, max_digits: int | None=_Unset, decimal_places: int | None=_Unset, min_length: int | None=_Unset, max_length: int | None=_Unset, union_mode: Literal['smart', 'left_to_right']=_Unset, **extra: Unpack[_EmptyKwargs]) -> Any:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/fields

    Create a field for objects that can be configured.

    Used to provide extra information about a field, either for the model schema or complex validation. Some arguments
    apply only to number fields (`int`, `float`, `Decimal`) and some apply only to `str`.

    Note:
        - Any `_Unset` objects will be replaced by the corresponding value defined in the `_DefaultValues` dictionary. If a key for the `_Unset` object is not found in the `_DefaultValues` dictionary, it will default to `None`

    Args:
        default: Default value if the field is not set.
        default_factory: A callable to generate the default value, such as :func:`~datetime.utcnow`.
        alias: The name to use for the attribute when validating or serializing by alias.
            This is often used for things like converting between snake and camel case.
        alias_priority: Priority of the alias. This affects whether an alias generator is used.
        validation_alias: Like `alias`, but only affects validation, not serialization.
        serialization_alias: Like `alias`, but only affects serialization, not validation.
        title: Human-readable title.
        description: Human-readable description.
        examples: Example values for this field.
        exclude: Whether to exclude the field from the model serialization.
        discriminator: Field name or Discriminator for discriminating the type in a tagged union.
        json_schema_extra: A dict or callable to provide extra JSON schema properties.
        frozen: Whether the field is frozen. If true, attempts to change the value on an instance will raise an error.
        validate_default: If `True`, apply validation to the default value every time you create an instance.
            Otherwise, for performance reasons, the default value of the field is trusted and not validated.
        repr: A boolean indicating whether to include the field in the `__repr__` output.
        init: Whether the field should be included in the constructor of the dataclass.
            (Only applies to dataclasses.)
        init_var: Whether the field should _only_ be included in the constructor of the dataclass.
            (Only applies to dataclasses.)
        kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
            (Only applies to dataclasses.)
        strict: If `True`, strict validation is applied to the field.
            See [Strict Mode](../concepts/strict_mode.md) for details.
        gt: Greater than. If set, value must be greater than this. Only applicable to numbers.
        ge: Greater than or equal. If set, value must be greater than or equal to this. Only applicable to numbers.
        lt: Less than. If set, value must be less than this. Only applicable to numbers.
        le: Less than or equal. If set, value must be less than or equal to this. Only applicable to numbers.
        multiple_of: Value must be a multiple of this. Only applicable to numbers.
        min_length: Minimum length for strings.
        max_length: Maximum length for strings.
        pattern: Pattern for strings (a regular expression).
        allow_inf_nan: Allow `inf`, `-inf`, `nan`. Only applicable to numbers.
        max_digits: Maximum number of allow digits for strings.
        decimal_places: Maximum number of decimal places allowed for numbers.
        union_mode: The strategy to apply when validating a union. Can be `smart` (the default), or `left_to_right`.
            See [Union Mode](standard_library_types.md#union-mode) for details.
        extra: (Deprecated) Extra fields that will be included in the JSON schema.

            !!! warning Deprecated
                The `extra` kwargs is deprecated. Use `json_schema_extra` instead.

    Returns:
        A new [`FieldInfo`][pydantic.fields.FieldInfo]. The return annotation is `Any` so `Field` can be used on
            type-annotated fields without causing a type error.
    """
    const = extra.pop('const', None)
    if const is not None:
        raise PydanticUserError('`const` is removed, use `Literal` instead', code='removed-kwargs')
    min_items = extra.pop('min_items', None)
    if min_items is not None:
        warn('`min_items` is deprecated and will be removed, use `min_length` instead', DeprecationWarning)
        if min_length in (None, _Unset):
            min_length = min_items
    max_items = extra.pop('max_items', None)
    if max_items is not None:
        warn('`max_items` is deprecated and will be removed, use `max_length` instead', DeprecationWarning)
        if max_length in (None, _Unset):
            max_length = max_items
    unique_items = extra.pop('unique_items', None)
    if unique_items is not None:
        raise PydanticUserError('`unique_items` is removed, use `Set` instead(this feature is discussed in https://github.com/pydantic/pydantic-core/issues/296)', code='removed-kwargs')
    allow_mutation = extra.pop('allow_mutation', None)
    if allow_mutation is not None:
        warn('`allow_mutation` is deprecated and will be removed. use `frozen` instead', DeprecationWarning)
        if allow_mutation is False:
            frozen = True
    regex = extra.pop('regex', None)
    if regex is not None:
        raise PydanticUserError('`regex` is removed. use `pattern` instead', code='removed-kwargs')
    if extra:
        warn(f'Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: {', '.join((k.__repr__() for k in extra.keys()))})', DeprecationWarning)
        if not json_schema_extra or json_schema_extra is _Unset:
            json_schema_extra = extra
    if validation_alias and validation_alias is not _Unset and (not isinstance(validation_alias, (str, AliasChoices, AliasPath))):
        raise TypeError('Invalid `validation_alias` type. it should be `str`, `AliasChoices`, or `AliasPath`')
    if serialization_alias in (_Unset, None) and isinstance(alias, str):
        serialization_alias = alias
    if validation_alias in (_Unset, None):
        validation_alias = alias
    include = extra.pop('include', None)
    if include is not None:
        warn('`include` is deprecated and does nothing. It will be removed, use `exclude` instead', DeprecationWarning)
    return FieldInfo.from_field(default, default_factory=default_factory, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, examples=examples, exclude=exclude, discriminator=discriminator, json_schema_extra=json_schema_extra, frozen=frozen, pattern=pattern, validate_default=validate_default, repr=repr, init=init, init_var=init_var, kw_only=kw_only, strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of, min_length=min_length, max_length=max_length, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, union_mode=union_mode)