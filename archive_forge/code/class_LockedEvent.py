from collections import defaultdict
from functools import partial
from threading import Lock
import inspect
import warnings
import logging
from transitions.core import Machine, Event, listify
class LockedEvent(Event):
    """ An event type which uses the parent's machine context map when triggered. """

    def trigger(self, model, *args, **kwargs):
        """ Extends transitions.core.Event.trigger by using locks/machine contexts. """
        if self.machine._ident.current != get_ident():
            with nested(*self.machine.model_context_map[id(model)]):
                return super(LockedEvent, self).trigger(model, *args, **kwargs)
        else:
            return super(LockedEvent, self).trigger(model, *args, **kwargs)