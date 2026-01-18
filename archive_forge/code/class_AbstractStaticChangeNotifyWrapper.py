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
class AbstractStaticChangeNotifyWrapper(object):
    """
    Concrete implementation must define the 'argument_transforms' class
    argument, a dictionary mapping the number of arguments in the event
    handler to a function that takes the arguments (obj, trait_name, old, new)
    and returns the arguments tuple for the actual handler.
    """
    arguments_transforms = {}

    def __init__(self, handler):
        arg_count = handler.__code__.co_argcount
        if arg_count > 4:
            raise TraitNotificationError('Invalid number of arguments for the static anytrait change notification handler: %s. A maximum of 4 arguments is allowed, but %s were specified.' % (handler.__name__, arg_count))
        self.argument_transform = self.argument_transforms[arg_count]
        self.handler = handler

    def __call__(self, object, trait_name, old, new):
        """ Dispatch to the appropriate handler method. """
        if _change_accepted(object, trait_name, old, new):
            args = self.argument_transform(object, trait_name, old, new)
            if _pre_change_event_tracer is not None:
                _pre_change_event_tracer(object, trait_name, old, new, self.handler)
            try:
                self.handler(*args)
            except Exception as e:
                if _post_change_event_tracer is not None:
                    _post_change_event_tracer(object, trait_name, old, new, self.handler, exception=e)
                handle_exception(object, trait_name, old, new)
            else:
                if _post_change_event_tracer is not None:
                    _post_change_event_tracer(object, trait_name, old, new, self.handler, exception=None)

    def equals(self, handler):
        return False