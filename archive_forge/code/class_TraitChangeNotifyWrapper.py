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
class TraitChangeNotifyWrapper(object):
    """ Dynamic change notify wrapper.

    This class is in charge to dispatch trait change events to dynamic
    listener, typically created using the `on_trait_change` method, or
    the decorator with the same name.
    """
    argument_transforms = {0: lambda obj, name, old, new: (), 1: lambda obj, name, old, new: (new,), 2: lambda obj, name, old, new: (name, new), 3: lambda obj, name, old, new: (obj, name, new), 4: lambda obj, name, old, new: (obj, name, old, new)}

    def __init__(self, handler, owner, target=None):
        self.init(handler, owner, target)

    def init(self, handler, owner, target=None):
        if type(handler) is MethodType:
            func = handler.__func__
            object = handler.__self__
            if object is not None:
                self.object = weakref.ref(object, self.listener_deleted)
                self.name = handler.__name__
                self.owner = owner
                arg_count = func.__code__.co_argcount - 1
                if arg_count > 4:
                    raise TraitNotificationError('Invalid number of arguments for the dynamic trait change notification handler: %s. A maximum of 4 arguments is allowed, but %s were specified.' % (func.__name__, arg_count))
                self.notify_listener = type(self)._notify_method_listener
                self.argument_transform = self.argument_transforms[arg_count]
                return arg_count
        elif target is not None:
            self.object = weakref.ref(target, self.listener_deleted)
            self.owner = owner
        arg_count = handler.__code__.co_argcount
        if arg_count > 4:
            raise TraitNotificationError('Invalid number of arguments for the dynamic trait change notification handler: %s. A maximum of 4 arguments is allowed, but %s were specified.' % (handler.__name__, arg_count))
        self.name = None
        self.handler = handler
        self.notify_listener = type(self)._notify_function_listener
        self.argument_transform = self.argument_transforms[arg_count]
        return arg_count

    def __call__(self, object, trait_name, old, new):
        """ Dispatch to the appropriate method.

        We do explicit dispatch instead of assigning to the .__call__ instance
        attribute to avoid reference cycles.
        """
        self.notify_listener(self, object, trait_name, old, new)

    def dispatch(self, handler, *args):
        """ Dispatch the event to the listener.

        This method is normally the only one that needs to be overridden in
        a subclass to implement the subclass's dispatch mechanism.
        """
        handler(*args)

    def equals(self, handler):
        if handler is self:
            return True
        if type(handler) is MethodType and handler.__self__ is not None:
            return handler.__name__ == self.name and handler.__self__ is self.object()
        return self.name is None and handler == self.handler

    def listener_deleted(self, ref):
        try:
            self.owner.remove(self)
        except ValueError:
            pass
        self.object = self.owner = None

    def dispose(self):
        self.object = None

    def _dispatch_change_event(self, object, trait_name, old, new, handler):
        """ Prepare and dispatch a trait change event to a listener. """
        args = self.argument_transform(object, trait_name, old, new)
        if _pre_change_event_tracer is not None:
            _pre_change_event_tracer(object, trait_name, old, new, handler)
        try:
            self.dispatch(handler, *args)
        except Exception as e:
            if _post_change_event_tracer is not None:
                _post_change_event_tracer(object, trait_name, old, new, handler, exception=e)
            handle_exception(object, trait_name, old, new)
        else:
            if _post_change_event_tracer is not None:
                _post_change_event_tracer(object, trait_name, old, new, handler, exception=None)

    def _notify_method_listener(self, object, trait_name, old, new):
        """ Dispatch a trait change event to a method listener. """
        obj_weak_ref = self.object
        if obj_weak_ref is not None and _change_accepted(object, trait_name, old, new):
            obj = obj_weak_ref()
            if obj is not None:
                listener = getattr(obj, self.name)
                self._dispatch_change_event(object, trait_name, old, new, listener)

    def _notify_function_listener(self, object, trait_name, old, new):
        """ Dispatch a trait change event to a function listener. """
        if _change_accepted(object, trait_name, old, new):
            self._dispatch_change_event(object, trait_name, old, new, self.handler)