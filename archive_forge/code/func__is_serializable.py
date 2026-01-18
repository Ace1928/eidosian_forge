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
def _is_serializable(value):
    """ Returns whether or not a specified value is serializable.
    """
    if isinstance(value, (list, tuple)):
        for item in value:
            if not _is_serializable(item):
                return False
        return True
    if isinstance(value, dict):
        for name, item in value.items():
            if not _is_serializable(name) or not _is_serializable(item):
                return False
        return True
    return not isinstance(value, HasTraits) or value.has_traits_interface(ISerializable)