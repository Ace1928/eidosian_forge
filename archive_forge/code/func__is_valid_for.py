import warnings
from .base_trait_handler import BaseTraitHandler
from .constants import ComparisonMode, DefaultValue, TraitKind
from .trait_base import Missing, Self, TraitsCache, Undefined, class_of
from .trait_dict_object import TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
def _is_valid_for(self, object, name, value):
    """ Handles a simplified validator that only returns whether or not the
            original value is valid.
        """
    if self.is_valid_for(value):
        return value
    self.error(object, name, value)