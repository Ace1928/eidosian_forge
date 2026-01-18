import warnings
from .base_trait_handler import BaseTraitHandler
from .constants import ComparisonMode, DefaultValue, TraitKind
from .trait_base import Missing, Self, TraitsCache, Undefined, class_of
from .trait_dict_object import TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
def _value_for(self, object, name, value):
    """ Handles a simplified validator that only receives the value
            argument.
        """
    try:
        return self.value_for(value)
    except TraitError:
        self.error(object, name, value)