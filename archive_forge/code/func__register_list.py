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
def _register_list(self, object, name, remove):
    """ Registers a handler for a list trait.
        """
    next = self.next
    if next is None:
        handler = self.handler()
        if handler is not Undefined:
            object._on_trait_change(handler, name, remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
            if self.is_list_handler:
                object._on_trait_change(self.handle_list_items_special, name + '_items', remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
            elif self.type == ANY_LISTENER:
                object._on_trait_change(handler, name + '_items', remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
        return (object, name)
    tl_handler = self.handle_list
    tl_handler_items = self.handle_list_items
    if self.notify:
        if self.type == DST_LISTENER:
            tl_handler = tl_handler_items = self.handle_error
        else:
            handler = self.handler()
            if handler is not Undefined:
                object._on_trait_change(handler, name, remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
                if self.is_list_handler:
                    object._on_trait_change(self.handle_list_items_special, name + '_items', remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
                elif self.type == ANY_LISTENER:
                    object._on_trait_change(handler, name + '_items', remove=remove, dispatch=self.dispatch, priority=self.priority, target=self._get_target())
    object._on_trait_change(tl_handler, name, remove=remove, dispatch='extended', priority=self.priority, target=self._get_target())
    object._on_trait_change(tl_handler_items, name + '_items', remove=remove, dispatch='extended', priority=self.priority, target=self._get_target())
    if remove:
        handler = next.unregister
    elif self.deferred:
        return INVALID_DESTINATION
    else:
        handler = next.register
    for obj in getattr(object, name):
        handler(obj)
    return INVALID_DESTINATION