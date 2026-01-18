import abc
import copy as copy_module
import inspect
import os
import pickle
import re
import types
import warnings
import weakref
from types import FunctionType
from . import __version__ as TraitsVersion
from .adaptation.adaptation_error import AdaptationError
from .constants import DefaultValue, TraitKind
from .ctrait import CTrait, __newobj__
from .ctraits import CHasTraits
from .observation import api as observe_api
from .traits import (
from .trait_types import Any, Bool, Disallow, Event, Python, Str
from .trait_notifiers import (
from .trait_base import (
from .trait_errors import TraitError
from .util.deprecated import deprecated
from .util._traitsui_helpers import check_traitsui_major_version
from .trait_converters import check_trait, mapped_trait_for, trait_for
@classmethod
def add_class_trait(cls, name, *trait):
    """ Adds a named trait attribute to this class.

        Also adds the same attribute to all subclasses.

        Parameters
        ----------
        name : str
            Name of the attribute to add.
        *trait :
            A trait or a value that can be converted to a trait using Trait()
            Trait definition of the attribute. It can be a single value or
            a list equivalent to an argument list for the Trait() function.

        """
    if len(trait) == 0:
        raise ValueError('No trait definition was specified.')
    if len(trait) > 1:
        trait = Trait(*trait)
    else:
        trait = trait_for(trait[0])
    cls._add_class_trait(name, trait, is_subclass=False)
    for subclass in cls.trait_subclasses(True):
        subclass._add_class_trait(name, trait, is_subclass=True)