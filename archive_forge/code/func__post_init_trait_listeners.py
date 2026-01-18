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
def _post_init_trait_listeners(self):
    """ Initializes the object's statically parsed, but dynamically
            registered, traits listeners (called at object creation and
            unpickling times).
        """
    for name, data in self.__class__.__listener_traits__.items():
        if data[0] == 'method':
            config = data[1]
            if config['post_init']:
                self.on_trait_change(getattr(self, name), config['pattern'], deferred=True, dispatch=config['dispatch'])