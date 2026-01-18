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
class PrefixMap(TraitType):
    """ A cross between the PrefixList and Map classes.

    Like Map, PrefixMap is created using a dictionary, but in this
    case, the keys of the dictionary must be strings. Like PrefixList,
    a string *v* is a valid value for the trait attribute if it is a prefix of
    one and only one key *k* in the dictionary. The actual values assigned to
    the trait attribute is *k*, and its corresponding mapped attribute is
    *map*[*k*].

    Example
    -------
    ::

        mapping = {'true': 1, 'yes': 1, 'false': 0, 'no': 0 }
        boolean_map = PrefixMap(mapping)

    This example defines a Boolean trait that accepts any prefix of 'true',
    'yes', 'false', or 'no', and maps them to 1 or 0.

    Parameters
    ----------
    map : dict
        A dictionary whose keys are strings that are valid values for the
        trait attribute, and whose corresponding values are the values for
        the shadow trait attribute.
    default_value : object, optional
        The default value for the trait. If given, this should be either a key
        from the mapping or a unique prefix of a key from the mapping. If not
        given, the first key from the mapping (in normal dictionary iteration
        order) will be used as the default.

    Attributes
    ----------
    map : dict
        A dictionary whose keys are strings that are valid values for the
        trait attribute, and whose corresponding values are the values for
        the shadow trait attribute.
    """
    default_value_type = DefaultValue.constant
    is_mapped = True

    def __init__(self, map, *, default_value=None, **metadata):
        map = dict(map)
        if not map:
            raise ValueError('map must be nonempty')
        self.map = map
        self._map = {value: value for value in map}
        if default_value is not None:
            default_value = self._complete_value(default_value)
        else:
            default_value = next(iter(self.map))
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

    def validate(self, object, name, value):
        if isinstance(value, str):
            try:
                return self._complete_value(value)
            except ValueError:
                pass
        self.error(object, name, value)

    def mapped_value(self, value):
        """ Get the mapped value for a value. """
        return self.map[value]

    def post_setattr(self, object, name, value):
        setattr(object, name + '_', self.mapped_value(value))

    def info(self):
        return ' or '.join((repr(x) for x in self.map)) + ' (or any unique prefix)'

    def get_editor(self, trait):
        from traitsui.api import EnumEditor
        return EnumEditor(values=self, cols=trait.cols or 3)