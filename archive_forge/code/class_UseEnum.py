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
class UseEnum(TraitType[t.Any, t.Any]):
    """Use a Enum class as model for the data type description.
    Note that if no default-value is provided, the first enum-value is used
    as default-value.

    .. sourcecode:: python

        # -- SINCE: Python 3.4 (or install backport: pip install enum34)
        import enum
        from traitlets import HasTraits, UseEnum


        class Color(enum.Enum):
            red = 1  # -- IMPLICIT: default_value
            blue = 2
            green = 3


        class MyEntity(HasTraits):
            color = UseEnum(Color, default_value=Color.blue)


        entity = MyEntity(color=Color.red)
        entity.color = Color.green  # USE: Enum-value (preferred)
        entity.color = "green"  # USE: name (as string)
        entity.color = "Color.green"  # USE: scoped-name (as string)
        entity.color = 3  # USE: number (as int)
        assert entity.color is Color.green
    """
    default_value: enum.Enum | None = None
    info_text = 'Trait type adapter to a Enum class'

    def __init__(self, enum_class: type[t.Any], default_value: t.Any=None, **kwargs: t.Any) -> None:
        assert issubclass(enum_class, enum.Enum), 'REQUIRE: enum.Enum, but was: %r' % enum_class
        allow_none = kwargs.get('allow_none', False)
        if default_value is None and (not allow_none):
            default_value = next(iter(enum_class.__members__.values()))
        super().__init__(default_value=default_value, **kwargs)
        self.enum_class = enum_class
        self.name_prefix = enum_class.__name__ + '.'

    def select_by_number(self, value: int, default: t.Any=Undefined) -> t.Any:
        """Selects enum-value by using its number-constant."""
        assert isinstance(value, int)
        enum_members = self.enum_class.__members__
        for enum_item in enum_members.values():
            if enum_item.value == value:
                return enum_item
        return default

    def select_by_name(self, value: str, default: t.Any=Undefined) -> t.Any:
        """Selects enum-value by using its name or scoped-name."""
        assert isinstance(value, str)
        if value.startswith(self.name_prefix):
            value = value.replace(self.name_prefix, '', 1)
        return self.enum_class.__members__.get(value, default)

    def validate(self, obj: t.Any, value: t.Any) -> t.Any:
        if isinstance(value, self.enum_class):
            return value
        elif isinstance(value, int):
            value2 = self.select_by_number(value)
            if value2 is not Undefined:
                return value2
        elif isinstance(value, str):
            value2 = self.select_by_name(value)
            if value2 is not Undefined:
                return value2
        elif value is None:
            if self.allow_none:
                return None
            else:
                return self.default_value
        self.error(obj, value)

    def _choices_str(self, as_rst: bool=False) -> str:
        """Returns a description of the trait choices (not none)."""
        choices = self.enum_class.__members__.keys()
        if as_rst:
            return '|'.join(('``%r``' % x for x in choices))
        else:
            return repr(list(choices))

    def _info(self, as_rst: bool=False) -> str:
        """Returns a description of the trait."""
        none = ' or %s' % ('`None`' if as_rst else 'None') if self.allow_none else ''
        return f'any of {self._choices_str(as_rst)}{none}'

    def info(self) -> str:
        return self._info(as_rst=False)

    def info_rst(self) -> str:
        return self._info(as_rst=True)