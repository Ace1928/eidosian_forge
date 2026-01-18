import warnings
from .base_trait_handler import BaseTraitHandler
from .constants import ComparisonMode, DefaultValue, TraitKind
from .trait_base import Missing, Self, TraitsCache, Undefined, class_of
from .trait_dict_object import TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
def _read_only(object, name, value):
    """ Raise a trait error for a read-only trait. """
    raise TraitError("The '%s' trait of %s instance is 'read only'." % (name, class_of(object)))