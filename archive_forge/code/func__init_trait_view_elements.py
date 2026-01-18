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
def _init_trait_view_elements(cls):
    """ Lazily Initialize the ViewElements object from a dictionary. """
    from traitsui.view_elements import ViewElements
    hastraits_bases = [base for base in cls.__bases__ if ClassTraits in base.__dict__]
    view_elements = ViewElements()
    elements_dict = cls.__dict__[ViewTraits]
    for name, element in elements_dict.items():
        if name in view_elements.content:
            raise TraitError("Duplicate definition for view element '%s'" % name)
        view_elements.content[name] = element
        element.replace_include(view_elements)
    for base in hastraits_bases:
        parent_view_elements = base.class_trait_view_elements()
        if parent_view_elements is not None:
            view_elements.parents.append(parent_view_elements)
    setattr(cls, ViewTraits, view_elements)
    return view_elements