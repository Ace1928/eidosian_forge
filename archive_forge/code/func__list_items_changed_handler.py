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
def _list_items_changed_handler(self, name, not_used, event):
    """ Handles adding/removing listeners for a generic 'List( Instance )'
            trait.
        """
    arg_lists = self._get_instance_handlers(name[:-6])
    for item in event.removed:
        for args in arg_lists:
            item.on_trait_change(*args, remove=True)
    for item in event.added:
        for args in arg_lists:
            item.on_trait_change(*args)