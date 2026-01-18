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
def __prefix_trait__(self, name, is_set):
    """ Return the trait definition for a specified name when there is
        no explicit definition in the class.
        """
    if name[:2] == '__' and name[-2:] == '__':
        if name == '__class__':
            return generic_trait
        if is_set:
            return any_trait
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
    if name[-1:] == '_':
        trait = self._trait(name[:-1], 0)
        if trait is not None and trait.type == 'delegate':
            return _clone_trait(trait)
    prefix_traits = self.__prefix_traits__
    for prefix in prefix_traits['*']:
        if prefix == name[:len(prefix)]:
            trait = prefix_traits[prefix]
            cls = self.__class__
            handlers = [_get_method(cls, '_%s_changed' % name), _get_method(cls, '_%s_fired' % name)]
            _add_event_handlers(trait, cls, handlers)
            handlers.append(prefix_traits.get('@'))
            handlers = [h for h in handlers if h is not None]
            if len(handlers) > 0:
                trait = _clone_trait(trait)
                _add_notifiers(trait._notifiers(True), handlers)
            return trait
    raise SystemError("Trait class look-up failed for attribute '%s' for an object of type '%s'" % (name, self.__class__.__name__))