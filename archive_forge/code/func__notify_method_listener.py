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
def _notify_method_listener(self, object, trait_name, old, new):
    """ Dispatch a trait change event to a method listener. """
    obj_weak_ref = self.object
    if obj_weak_ref is not None:
        obj = obj_weak_ref()
        if obj is not None:
            listener = getattr(obj, self.name)
            self._dispatch_change_event(object, trait_name, old, new, listener)