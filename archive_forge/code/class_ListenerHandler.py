import re
import string
import weakref
from string import whitespace
from types import MethodType
from .constants import DefaultValue
from .trait_base import Undefined, Uninitialized
from .trait_errors import TraitError
from .trait_notifiers import TraitChangeNotifyWrapper
from .util.weakiddict import WeakIDKeyDict
class ListenerHandler(object):
    """
    Wrapper for trait change handlers that avoids strong references to methods.

    For a bound method handler, this wrapper prevents us from holding a
    strong reference to the object bound to that bound method. For other
    callable handlers, we do keep a strong reference to the handler.

    When called with no arguments, this object returns either the actual
    handler, or Undefined if the handler no longer exists because the object
    it was bound to has been garbage collected.

    Parameters
    ----------
    handler : callable
        Object to be called when the relevant trait or traits change.
    """

    def __init__(self, handler):
        if isinstance(handler, MethodType):
            self.handler_ref = weakref.WeakMethod(handler)
        else:
            self.handler = handler

    def __call__(self):
        result = getattr(self, 'handler', None)
        if result is not None:
            return result
        else:
            handler = self.handler_ref()
            return Undefined if handler is None else handler