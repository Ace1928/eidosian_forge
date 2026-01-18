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
class _NoneTrait(TraitType):
    """ Defines a trait that only accepts the None value

    This is primarily used for supporting ``Union``.
    """
    info_text = 'None'
    default_value = None
    default_value_type = DefaultValue.constant

    def __init__(self, **metadata):
        default_value = metadata.pop('default_value', None)
        if default_value is not None:
            raise ValueError('Cannot set default value {} for _NoneTrait'.format(default_value))
        super().__init__(**metadata)

    def validate(self, obj, name, value):
        if value is None:
            return value
        self.error(obj, name, value)