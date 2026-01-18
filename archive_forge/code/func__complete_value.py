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
            Equal to *value*, if *value* is already a member of self.map.
            Otherwise, the unique member of self.values for which *value*
            is a prefix.

        Raises
        ------
        ValueError
            If value is not in self.map, and is not a prefix of any
            element of self.map, or is a prefix of multiple elements
            of self.map.
        """
    if value in self.map:
        return value
    matches = [key for key in self.map if key.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f'{value!r} is neither a member nor a unique prefix of a member of {list(self.map)}')