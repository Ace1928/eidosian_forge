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
@classmethod
def _add_class_trait(cls, name, trait, is_subclass):
    """
        Add a named trait attribute to this class.

        Does not affect subclasses.

        Parameters
        ----------
        name : str
            Name of the attribute to add.
        trait : CTrait
            The trait to be added.
        is_subclass : bool
            True if we're adding the trait to a strict subclass of the
            original class that add_class_trait was called for. This is used
            to decide how to behave if ``cls`` already has a trait named
            ``name``: in that circumstance, if ``is_subclass`` is False, an
            error will be raised, while if ``is_subclass`` is True, no trait
            will be added.

        Raises
        ------
        TraitError
            If a trait with the given name already exists, and is_subclass
            is ``False``.
        """
    class_dict = cls.__dict__
    prefix_traits = class_dict[PrefixTraits]
    if name[-1:] == '_':
        name = name[:-1]
        if name in prefix_traits:
            if is_subclass:
                return
            raise TraitError("The '%s_' trait is already defined." % name)
        prefix_traits[name] = trait
        prefix_list = prefix_traits['*']
        prefix_list.append(name)
        prefix_list.sort(key=len, reverse=True)
        return
    class_traits = class_dict[ClassTraits]
    if class_traits.get(name) is not None:
        if is_subclass:
            return
        raise TraitError("The '%s' trait is already defined." % name)
    handler = trait.handler
    if handler is not None:
        if handler.has_items:
            cls._add_class_trait(name + '_items', handler.items_event(), is_subclass=is_subclass)
        if handler.is_mapped:
            cls._add_class_trait(name + '_', mapped_trait_for(trait, name), is_subclass=is_subclass)
    if trait.is_base is not False:
        class_dict[BaseTraits][name] = trait
    handlers = [_get_method(cls, '_%s_changed' % name), _get_method(cls, '_%s_fired' % name)]
    _add_event_handlers(trait, cls, handlers)
    handlers.append(prefix_traits.get('@'))
    handlers = [h for h in handlers if h is not None]
    if len(handlers) > 0:
        trait = _clone_trait(trait)
        _add_notifiers(trait._notifiers(True), handlers)
    class_traits[name] = trait