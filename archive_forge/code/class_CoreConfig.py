from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class CoreConfig(TypedDict, total=False):
    """
    Base class for schema configuration options.

    Attributes:
        title: The name of the configuration.
        strict: Whether the configuration should strictly adhere to specified rules.
        extra_fields_behavior: The behavior for handling extra fields.
        typed_dict_total: Whether the TypedDict should be considered total. Default is `True`.
        from_attributes: Whether to use attributes for models, dataclasses, and tagged union keys.
        loc_by_alias: Whether to use the used alias (or first alias for "field required" errors) instead of
            `field_names` to construct error `loc`s. Default is `True`.
        revalidate_instances: Whether instances of models and dataclasses should re-validate. Default is 'never'.
        validate_default: Whether to validate default values during validation. Default is `False`.
        populate_by_name: Whether an aliased field may be populated by its name as given by the model attribute,
            as well as the alias. (Replaces 'allow_population_by_field_name' in Pydantic v1.) Default is `False`.
        str_max_length: The maximum length for string fields.
        str_min_length: The minimum length for string fields.
        str_strip_whitespace: Whether to strip whitespace from string fields.
        str_to_lower: Whether to convert string fields to lowercase.
        str_to_upper: Whether to convert string fields to uppercase.
        allow_inf_nan: Whether to allow infinity and NaN values for float fields. Default is `True`.
        ser_json_timedelta: The serialization option for `timedelta` values. Default is 'iso8601'.
        ser_json_bytes: The serialization option for `bytes` values. Default is 'utf8'.
        ser_json_inf_nan: The serialization option for infinity and NaN values
            in float fields. Default is 'null'.
        hide_input_in_errors: Whether to hide input data from `ValidationError` representation.
        validation_error_cause: Whether to add user-python excs to the __cause__ of a ValidationError.
            Requires exceptiongroup backport pre Python 3.11.
        coerce_numbers_to_str: Whether to enable coercion of any `Number` type to `str` (not applicable in `strict` mode).
        regex_engine: The regex engine to use for regex pattern validation. Default is 'rust-regex'. See `StringSchema`.
    """
    title: str
    strict: bool
    extra_fields_behavior: ExtraBehavior
    typed_dict_total: bool
    from_attributes: bool
    loc_by_alias: bool
    revalidate_instances: Literal['always', 'never', 'subclass-instances']
    validate_default: bool
    populate_by_name: bool
    str_max_length: int
    str_min_length: int
    str_strip_whitespace: bool
    str_to_lower: bool
    str_to_upper: bool
    allow_inf_nan: bool
    ser_json_timedelta: Literal['iso8601', 'float']
    ser_json_bytes: Literal['utf8', 'base64', 'hex']
    ser_json_inf_nan: Literal['null', 'constants']
    hide_input_in_errors: bool
    validation_error_cause: bool
    coerce_numbers_to_str: bool
    regex_engine: Literal['rust-regex', 'python-re']