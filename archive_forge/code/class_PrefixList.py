import collections.abc
import datetime
from importlib import import_module
import operator
from os import fspath
from os.path import isfile, isdir
import re
import sys
from types import FunctionType, MethodType, ModuleType
import uuid
import warnings
from .constants import DefaultValue, TraitKind, ValidateTrait
from .trait_base import (
from .trait_converters import trait_from, trait_cast
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListEvent, TraitListObject
from .trait_set_object import TraitSetEvent, TraitSetObject
from .trait_type import (
from .traits import (
from .util.deprecated import deprecated
from .util.import_symbol import import_symbol
from .editor_factories import (
class PrefixList(TraitType):
    """Ensures that a value assigned to the attribute is a member of a list of
     specified string values, or is a unique prefix of one of those values.

    The values that can be assigned to a trait attribute of type PrefixList
    type is the set of all strings supplied to the PrefixList constructor,
    as well as any unique prefix of those strings. That is, if the set of
    strings supplied to the constructor is described by
    [*s*\\ :sub:`1`\\ , *s*\\ :sub:`2`\\ , ..., *s*\\ :sub:`n`\\ ], then the
    string *v* is a valid value for the trait if *v* == *s*\\ :sub:`i[:j]`
    for one and only one pair of values (i, j). If *v* is a valid value,
    then the actual value assigned to the trait attribute is the
    corresponding *s*\\ :sub:`i` value that *v* matched.

    The legal values can be provided as an iterable of values.

    Example
    -------
    ::
        class Person(HasTraits):
            married = PrefixList(['yes', 'no'])

    The Person class has a **married** trait that accepts any of the
    strings 'y', 'ye', 'yes', 'n', or 'no' as valid values. However, the
    actual values assigned as the value of the trait attribute are limited
    to either 'yes' or 'no'. That is, if the value 'y' is assigned to the
    **married** attribute, the actual value assigned will be 'yes'.

    Parameters
    ----------
    values
        A list or other iterable of legal string values for this trait.

    Attributes
    ----------
    values : list of str
        The list of legal values for this trait.
    """
    default_value_type = DefaultValue.constant

    def __init__(self, values, *, default_value=None, **metadata):
        if isinstance(values, (str, bytes, bytearray)):
            raise TypeError(f'values should be a collection of strings, not {values!r}')
        values = list(values)
        if not values:
            raise ValueError('values must be nonempty')
        self.values = values
        self._values_as_set = frozenset(values)
        if default_value is not None:
            default_value = self._complete_value(default_value)
        else:
            default_value = self.values[0]
        super().__init__(default_value, **metadata)

    def _complete_value(self, value):
        """
        Validate and complete a given value.

        Parameters
        ----------
        value : str
            Value to be validated.

        Returns
        -------
        completion : str
            Equal to *value*, if *value* is already a member of self.values.
            Otherwise, the unique member of self.values for which *value*
            is a prefix.

        Raises
        ------
        ValueError
            If value is not in self.values, and is not a prefix of any
            element of self.values, or is a prefix of multiple elements
            of self.values.
        """
        if value in self._values_as_set:
            return value
        matches = [key for key in self.values if key.startswith(value)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f'{value!r} is neither a member nor a unique prefix of a member of {self.values}')

    def validate(self, object, name, value):
        if isinstance(value, str):
            try:
                return self._complete_value(value)
            except ValueError:
                pass
        self.error(object, name, value)

    def info(self):
        return ' or '.join((repr(x) for x in self.values)) + ' (or any unique prefix)'