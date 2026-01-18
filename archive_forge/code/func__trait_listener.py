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
def _trait_listener(self, object, prefix, remove):
    if prefix[-1:] != '_':
        prefix += '_'
    n = len(prefix)
    traits = self.__base_traits__
    for name in self._each_trait_method(object):
        if name[:n] == prefix:
            if name[-8:] == '_changed':
                short_name = name[n:-8]
                if short_name in traits:
                    self._on_trait_change(getattr(object, name), short_name, remove=remove)
                elif short_name == 'anytrait':
                    self._on_trait_change(getattr(object, name), remove=remove)
            elif name[:-6] == '_fired':
                short_name = name[n:-6]
                if short_name in traits:
                    self._on_trait_change(getattr(object, name), short_name, remove=remove)
                elif short_name == 'anytrait':
                    self._on_trait_change(getattr(object, name), remove=remove)