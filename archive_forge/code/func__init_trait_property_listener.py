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
def _init_trait_property_listener(self, name, kind, cached, pattern):
    """ Sets up the listener for a property with 'depends_on' metadata.
        """
    if cached is None:

        @weak_arg(self)
        def notify(self):
            self.trait_property_changed(name, None)
    else:
        cached_old = cached + ':old'

        @weak_arg(self)
        def pre_notify(self):
            dict = self.__dict__
            old = dict.get(cached_old, Undefined)
            if old is Undefined:
                dict[cached_old] = dict.pop(cached, None)
        self.on_trait_change(pre_notify, pattern, priority=True, target=self)

        @weak_arg(self)
        def notify(self):
            old = self.__dict__.pop(cached_old, Undefined)
            if old is not Undefined:
                self.trait_property_changed(name, old)
    self.on_trait_change(notify, pattern, target=self)