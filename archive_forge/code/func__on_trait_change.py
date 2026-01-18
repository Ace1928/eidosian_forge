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
def _on_trait_change(self, handler, name=None, remove=False, dispatch='same', priority=False, target=None):
    """Causes the object to invoke a handler whenever a trait attribute
        is modified, or removes the association.

        Multiple handlers can be defined for the same object, or even for the
        same trait attribute on the same object. If *name* is not specified or
        is None, *handler* is invoked when any trait attribute on the
        object is changed.

        Parameters
        ----------
        handler : function
            A trait notification function for the attribute specified by
            *name*.
        name : str
            Specifies the trait attribute whose value changes trigger the
            notification.
        remove : bool
            If True, removes the previously-set association between
            *handler* and *name*; if False (the default), creates the
            association.
        """
    if type(name) is list:
        for name_i in name:
            self._on_trait_change(handler, name_i, remove, dispatch, priority, target)
        return
    name = name or 'anytrait'
    if remove:
        if name == 'anytrait':
            notifiers = self._notifiers(False)
        else:
            trait = self._trait(name, 1)
            if trait is None:
                return
            notifiers = trait._notifiers(False)
        if notifiers is not None:
            for i, notifier in enumerate(notifiers):
                if notifier.equals(handler):
                    del notifiers[i]
                    notifier.dispose()
                    break
        return
    if name == 'anytrait':
        notifiers = self._notifiers(True)
    else:
        notifiers = self._trait(name, 2)._notifiers(True)
    for notifier in notifiers:
        if notifier.equals(handler):
            break
    else:
        wrapper = self.wrappers[dispatch](handler, notifiers, target)
        if priority:
            notifiers.insert(0, wrapper)
        else:
            notifiers.append(wrapper)