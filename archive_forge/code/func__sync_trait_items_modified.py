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
def _sync_trait_items_modified(self, object, name, old, event):
    n0 = event.index
    n1 = n0 + len(event.removed)
    name = name[:-6]
    info = self.__sync_trait__
    locked = info['']
    locked[name] = None
    for object, object_name in info[name].values():
        object = object()
        if object_name not in object._get_sync_trait_info()['']:
            try:
                getattr(object, object_name)[n0:n1] = event.added
            except:
                pass
    del locked[name]