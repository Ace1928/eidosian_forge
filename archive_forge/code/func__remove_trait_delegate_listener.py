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
def _remove_trait_delegate_listener(self, name, remove):
    """ Removes a delegate listener when the local delegate value is set.
        """
    dict = self.__dict__.setdefault(ListenerTraits, {})
    if remove:
        if name in dict:
            self.on_trait_change(dict[name], self._trait_delegate_name(name, self.__class__.__listener_traits__[name][1]), remove=True)
            del dict[name]
            if len(dict) == 0:
                del self.__dict__[ListenerTraits]
        return
    if name not in dict:
        self._init_trait_delegate_listener(name, 0, self.__class__.__listener_traits__[name][1])