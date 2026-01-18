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
def _register_simple(self, object, name, remove):
    """ Registers a handler for a simple trait.
        """
    next = self.next
    if next is None:
        handler = self.handler()
        if handler is not Undefined:
            object._on_trait_change(handler, name, remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
        return (object, name)
    tl_handler = self.handle_simple
    if self.notify:
        if self.type == DST_LISTENER:
            if self.dispatch != 'same':
                raise TraitError("Trait notification dispatch type '%s' is not compatible with handler signature and extended trait name notification style" % self.dispatch)
            tl_handler = self.handle_dst
        else:
            handler = self.handler()
            if handler is not Undefined:
                object._on_trait_change(handler, name, remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
    object._on_trait_change(tl_handler, name, remove=remove, dispatch='extended', priority=self.priority, target=self._get_target())
    if remove:
        return next.unregister(getattr(object, name))
    if not self.deferred or name in object.__dict__:
        return next.register(getattr(object, name))
    return (object, name)