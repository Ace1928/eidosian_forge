import contextlib
import logging
import threading
from threading import local as thread_local
from threading import Thread
import traceback
from types import MethodType
import weakref
import sys
from .constants import ComparisonMode, TraitKind
from .trait_base import Uninitialized
from .trait_errors import TraitNotificationError
def _change_accepted(object, name, old, new):
    """ Return true if notifications should be emitted for the change.

    Parameters
    ----------
    object : HasTraits
        The object on which the trait is changed.
    name : str
        The name of the trait changed.
    old : any
        The old value
    new : any
        The new value

    Returns
    -------
    accepted : bool
        Whether the event should be emitted.
    """
    if old is Uninitialized:
        return False
    trait = object._trait(name, 2)
    if trait.type == TraitKind.trait.name and trait.comparison_mode == ComparisonMode.equality:
        try:
            return bool(old != new)
        except Exception:
            pass
    return True