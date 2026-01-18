import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
class _TraitsChangeCollector(HasStrictTraits):
    """ Class allowing thread-safe recording of events.
    """
    obj = Any
    trait_name = Str
    event_count = Property(Int)
    event_count_updated = Event
    events = List(Any)
    _lock = Any()

    def __init__(self, **traits):
        if 'trait' in traits:
            value = traits.pop('trait')
            message = 'The `trait` keyword is deprecated. please use `trait_name`'
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            traits['trait_name'] = value
        super().__init__(**traits)
        self._lock = threading.Lock()
        self.events = []

    def start_collecting(self):
        self.obj.on_trait_change(self._event_handler, self.trait_name)

    def stop_collecting(self):
        self.obj.on_trait_change(self._event_handler, self.trait_name, remove=True)

    def _event_handler(self, new):
        with self._lock:
            self.events.append(new)
        self.event_count_updated = True

    def _get_event_count(self):
        """ Traits property getter.

        Thread-safe access to event count.

        """
        with self._lock:
            return len(self.events)