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
class ExtendedTraitChangeNotifyWrapper(TraitChangeNotifyWrapper):
    """ Change notify wrapper for "extended" trait change events..

    The "extended notifiers" are set up internally when using extended traits,
    to add/remove traits listeners when one of the intermediate traits changes.

    For example, in a listener for the extended trait `a.b`, we need to
    add/remove listeners to `a:b` when `a` changes.
    """

    def _dispatch_change_event(self, object, trait_name, old, new, handler):
        """ Prepare and dispatch a trait change event to a listener. """
        args = self.argument_transform(object, trait_name, old, new)
        try:
            self.dispatch(handler, *args)
        except Exception:
            handle_exception(object, trait_name, old, new)

    def _notify_method_listener(self, object, trait_name, old, new):
        """ Dispatch a trait change event to a method listener. """
        obj_weak_ref = self.object
        if obj_weak_ref is not None:
            obj = obj_weak_ref()
            if obj is not None:
                listener = getattr(obj, self.name)
                self._dispatch_change_event(object, trait_name, old, new, listener)

    def _notify_function_listener(self, object, trait_name, old, new):
        """ Dispatch a trait change event to a function listener. """
        self._dispatch_change_event(object, trait_name, old, new, self.handler)