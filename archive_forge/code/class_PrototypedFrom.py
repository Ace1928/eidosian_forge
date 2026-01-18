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
class PrototypedFrom(Delegate):
    """ A trait type that matches the 'prototype' design pattern.

    This defines a trait whose default value and definition is "prototyped"
    from another trait on a different object.

    An object containing a prototyped trait attribute must contain a
    second attribute that references the object containing the prototype
    trait attribute. The name of this second attribute is passed as the
    *prototype* argument to the PrototypedFrom() function.

    The following rules govern the application of the prefix parameter:

    * If *prefix* is empty or omitted, the prototype delegation is to an
      attribute of the prototype object with the same name as the
      prototyped attribute.
    * If *prefix* is a valid Python attribute name, then the prototype
      delegation is to an attribute whose name is the value of *prefix*.
    * If *prefix* ends with an asterisk ('*') and is longer than one
      character, then the prototype delegation is to an attribute whose
      name is the value of *prefix*, minus the trailing asterisk,
      prepended to the prototyped attribute name.
    * If *prefix* is equal to a single asterisk, the prototype
      delegation is to an attribute whose name is the value of the
      prototype object's __prefix__ attribute prepended to the
      prototyped attribute name.

    Note that any changes to the prototyped attribute are made to the
    original object, not the prototype object. The prototype object is
    only used to define to trait type and default value.

    Parameters
    ----------
    prototype : str
        Name of the attribute on the current object which references the
        object that is the trait's prototype.
    prefix : str
        A prefix or substitution applied to the original attribute when
        looking up the prototyped attribute.
    listenable : bool
        Indicates whether a listener can be attached to this attribute
        such that changes to the corresponding attribute on the
        prototype object will trigger it.
    **metadata
        Trait metadata for the trait.
    """

    def __init__(self, prototype, prefix='', listenable=True, **metadata):
        super().__init__(prototype, prefix=prefix, modify=False, listenable=listenable, **metadata)