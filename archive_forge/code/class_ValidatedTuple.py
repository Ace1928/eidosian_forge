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
class ValidatedTuple(BaseTuple):
    """ A trait type holding a tuple with customized validation.

    Parameters
    ----------
    *types
        Definition of the default and allowed tuples. (see
        :class:`~.BaseTuple` for more details)
    fvalidate : callable, optional
        A callable to provide the additional custom validation for the
        tuple. The callable will be passed the tuple value and should
        return True or False.
    fvalidate_info : string, optional
        A string describing the custom validation to use for the error
        messages.
    **metadata
        Trait metadata for the trait.

    Example
    -------
    The definition::

        value_range = ValidatedTuple(
            Int(0), Int(1), fvalidate=lambda x: x[0] < x[1])

    will accept only tuples ``(a, b)`` containing two integers that
    satisfy ``a < b``.
    """

    def __init__(self, *types, **metadata):
        metadata.setdefault('fvalidate', None)
        metadata.setdefault('fvalidate_info', '')
        super().__init__(*types, **metadata)

    def validate(self, object, name, value):
        """ Validates that the value is a valid tuple.
        """
        values = super().validate(object, name, value)
        if self.fvalidate is None or self.fvalidate(values):
            return values
        else:
            self.error(object, name, value)

    def full_info(self, object, name, value):
        """ Returns a description of the trait.
        """
        message = 'a tuple of the form: ({0}) that passes custom validation{1}'
        types_info = ', '.join([type_.full_info(object, name, value) for type_ in self.types])
        if self.fvalidate_info is not None:
            fvalidate_info = ': {0}'.format(self.fvalidate_info)
        else:
            fvalidate_info = ''
        return message.format(types_info, fvalidate_info)