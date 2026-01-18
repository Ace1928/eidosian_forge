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
class WeakRef(Instance):
    """ A trait type holding a weak reference to an instance of a class.

    Only a weak reference is maintained to any object assigned to a WeakRef
    trait. If no other references exist to the assigned value, the value
    may be garbage collected, in which case the value of the trait becomes
    None. In all other cases, the value returned by the trait is the
    original object.

    Parameters
    ----------
    klass : class, str or instance
        The object that forms the basis for the trait. If *klass* is
        omitted, then values must be an instance of HasTraits.  If a string,
        the value will be resolved to a class object at runtime.
    allow_none : boolean
        Indicates whether None can be _assigned_.  The trait attribute may
        give a None value if the object referred to has been garbage collected
        even if allow_none is False.
    adapt : str
        How to use the adaptation infrastructure when setting the value.
    """

    def __init__(self, klass='traits.has_traits.HasTraits', allow_none=False, adapt='yes', **metadata):
        metadata.setdefault('copy', 'ref')
        super().__init__(klass, allow_none=allow_none, adapt=adapt, module=get_module_name(), **metadata)

    def get(self, object, name):
        value = getattr(object, name + '_', None)
        if value is not None:
            return value.value()
        return None

    def set(self, object, name, value):
        old = self.get(object, name)
        if value is None:
            object.__dict__[name + '_'] = None
        else:
            object.__dict__[name + '_'] = HandleWeakRef(object, name, value)
        if value is not old:
            object.trait_property_changed(name, old, value)

    def resolve_class(self, object, name, value):
        klass = self.find_class(self.klass)
        if klass is None:
            self.validate_failed(object, name, value)
        self.klass = klass