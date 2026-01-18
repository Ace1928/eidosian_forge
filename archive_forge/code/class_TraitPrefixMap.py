from importlib import import_module
import sys
from types import FunctionType, MethodType
from .constants import DefaultValue, ValidateTrait
from .trait_base import (
from .trait_base import RangeTypes  # noqa: F401, used by TraitsUI
from .trait_errors import TraitError
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_converters import trait_from
from .trait_handler import TraitHandler
from .trait_list_object import TraitListEvent, TraitListObject
from .util.deprecated import deprecated
class TraitPrefixMap(TraitMap):
    """A cross between the TraitPrefixList and TraitMap classes.

    Like TraitMap, TraitPrefixMap is created using a dictionary, but in this
    case, the keys of the dictionary must be strings. Like TraitPrefixList,
    a string *v* is a valid value for the trait attribute if it is a prefix of
    one and only one key *k* in the dictionary. The actual values assigned to
    the trait attribute is *k*, and its corresponding mapped attribute is
    *map*[*k*].

    Example
    -------
    ::

        mapping = {'true': 1, 'yes': 1, 'false': 0, 'no': 0 }
        boolean_map = Trait('true', TraitPrefixMap(mapping))

    This example defines a Boolean trait that accepts any prefix of 'true',
    'yes', 'false', or 'no', and maps them to 1 or 0.

    Parameters
    ----------
    map : dict
        A dictionary whose keys are strings that are valid values for the
        trait attribute, and whose corresponding values are the values for
        the shadow trait attribute.

    Attributes
    ----------
    map : dict
        A dictionary whose keys are strings that are valid values for the
        trait attribute, and whose corresponding values are the values for
        the shadow trait attribute.
    """

    @deprecated(_WARNING_FORMAT_STR.format(handler='TraitPrefixMap', replacement='PrefixMap'))
    def __init__(self, map):
        self.map = map
        self._map = _map = {}
        for key in map.keys():
            _map[key] = key
        self.fast_validate = (ValidateTrait.prefix_map, _map, self.validate)

    def validate(self, object, name, value):
        try:
            if value not in self._map:
                match = None
                n = len(value)
                for key in self.map.keys():
                    if value == key[:n]:
                        if match is not None:
                            match = None
                            break
                        match = key
                if match is None:
                    self.error(object, name, value)
                self._map[value] = match
            return self._map[value]
        except:
            self.error(object, name, value)

    def info(self):
        return super().info() + ' (or any unique prefix)'