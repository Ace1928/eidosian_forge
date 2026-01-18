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
class TraitDict(TraitHandler):
    """ Ensures that values assigned to a trait attribute are dictionaries
    whose keys and values are of specified types.

    TraitDict also makes sure that any changes to keys or values made that are
    made after the dictionary is assigned to the trait attribute satisfy the
    type constraints. TraitDict is the underlying handler for the
    dictionary-based predefined traits, and the Dict() trait factory.

    Example
    -------
    ::

        class WorkoutClass(HasTraits):
            member_weights = Trait({}, TraitDict(str, float))


    This example defines a WorkoutClass class containing a *member_weights*
    trait attribute whose value must be a dictionary containing keys that
    are strings (i.e., the members' names) and whose associated values must
    be floats (i.e., their most recently recorded weight).

    Parameters
    ----------
    key_trait : trait
        The type for the dictionary keys.  If this is None or omitted, the
        keys in the dictionary can be of any type. Otherwise, this
        must be either a trait, or a value that can be converted to a trait
        using the trait_from() function. In this case, all dictionary keys are
        checked to ensure that they are of the type specified.
    value_trait : trait
        The type for the dictionary values.  If this is None or omitted, the
        values in the dictionary can be of any type. Otherwise, this must be
        either a trait, or a value that can be converted to a trait using the
        trait_from() function.  In this case, all dictionary values are
        checked to ensure that they are of the type specified.
    has_items : bool
        Flag indicating whether the dictionary contains entries.

    Attributes
    ----------
    key_trait : CTrait or TraitHandler or None
        The type for the dictionary keys.  If this is None then the keys are
        not validated.
    value_trait : CTrait or TraitHandler or None
        The type for the dictionary values.  If this is None then the values
        are not validated.
    value_handler : BaseTraitHandler or None
        The trait handler for the dictionary values.
    has_items : bool
        Flag indicating whether the dictionary contains entries.
    """
    info_trait = None
    default_value_type = DefaultValue.trait_list_object
    _items_event = None

    @deprecated(_WARNING_FORMAT_STR.format(handler='TraitDict', replacement='Dict'))
    def __init__(self, key_trait=None, value_trait=None, has_items=True):
        self.key_trait = trait_from(key_trait)
        self.value_trait = trait_from(value_trait)
        self.has_items = has_items
        handler = self.value_trait.handler
        if handler.has_items:
            handler = handler.clone()
            handler.has_items = False
        self.value_handler = handler

    def clone(self):
        return TraitDict(self.key_trait, self.value_trait, self.has_items)

    def validate(self, object, name, value):
        if isinstance(value, dict):
            return TraitDictObject(self, object, name, value)
        self.error(object, name, value)

    def full_info(self, object, name, value):
        extra = ''
        handler = self.key_trait.handler
        if handler is not None:
            extra = ' with keys which are %s' % handler.full_info(object, name, value)
        handler = self.value_handler
        if handler is not None:
            if extra == '':
                extra = ' with'
            else:
                extra += ' and'
            extra += ' values which are %s' % handler.full_info(object, name, value)
        return 'a dictionary%s' % extra

    def get_editor(self, trait):
        if self.editor is None:
            from traitsui.api import TextEditor
            self.editor = TextEditor(evaluate=eval)
        return self.editor

    def items_event(self):
        from .trait_types import Event
        if TraitDict._items_event is None:
            TraitDict._items_event = Event(TraitDictEvent, is_base=False).as_ctrait()
        return TraitDict._items_event