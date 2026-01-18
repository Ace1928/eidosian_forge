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
def _create_property_observe_state(observe, property_name, cached):
    """ Create the metadata for setting up an observer for Property.

    Parameters
    ----------
    observe : str or list or Expression
        As is accepted by HasTraits.observe expression argument
        This is the value provided in Property(observe=...)
    property_name : str
        The name of the property trait.
    cached : boolean
        Whether the property is cached or not.

    Returns
    -------
    state : dict
        State to be used by _init_traits_observers
    """

    def handler(instance, event):
        if cached:
            cache_name = TraitsCache + property_name
            old = instance.__dict__.pop(cache_name, Undefined)
        else:
            old = Undefined
        instance.trait_property_changed(property_name, old)

    def handler_getter(instance, name):
        return types.MethodType(handler, instance)
    graphs = _compile_expression(observe)
    return dict(graphs=graphs, dispatch='same', handler_getter=handler_getter, post_init=False)