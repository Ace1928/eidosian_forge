import warnings
from .base_trait_handler import BaseTraitHandler
from .constants import ComparisonMode, DefaultValue, TraitKind
from .trait_base import Missing, Self, TraitsCache, Undefined, class_of
from .trait_dict_object import TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
def _infer_default_value_type(default_value):
    """ Figure out the default value type given a default value.
    """
    if default_value is Missing:
        return DefaultValue.missing
    elif default_value is Self:
        return DefaultValue.object
    elif isinstance(default_value, TraitListObject):
        return DefaultValue.trait_list_object
    elif isinstance(default_value, TraitDictObject):
        return DefaultValue.trait_dict_object
    elif isinstance(default_value, TraitSetObject):
        return DefaultValue.trait_set_object
    elif isinstance(default_value, list):
        return DefaultValue.list_copy
    elif isinstance(default_value, dict):
        return DefaultValue.dict_copy
    else:
        return DefaultValue.constant