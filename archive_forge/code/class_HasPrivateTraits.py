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
class HasPrivateTraits(HasTraits):
    """ This class ensures that any public object attribute that does not have
    an explicit or wildcard trait definition results in an exception, but
    "private" attributes (whose names start with '_') have an initial value of
    **None**, and are not type-checked.

    This feature is useful in cases where a class needs private attributes to
    keep track of its internal object state, which are not part of the class's
    public API. Such attributes do not need to be type-checked, because they
    are manipulated only by the (presumably correct) methods of the class
    itself.
    """
    __ = Any(private=True, transient=True)
    _ = Disallow