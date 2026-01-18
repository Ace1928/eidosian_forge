import os
import logging
import unicodedata
from threading import Thread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.events import (
from wandb_watchdog.observers.api import (
import AppKit
from FSEvents import (
from FSEvents import (
class FSEventsEmitter(EventEmitter):
    """
    FSEvents based event emitter. Handles conversion of native events.
    """

    def __init__(self, event_queue, watch, timeout=DEFAULT_EMITTER_TIMEOUT):
        EventEmitter.__init__(self, event_queue, watch, timeout)
        self._fsevents = FSEventsQueue(watch.path)
        self._fsevents.start()

    def on_thread_stop(self):
        self._fsevents.stop()

    def queue_events(self, timeout):
        events = self._fsevents.read_events()
        if events is None:
            return
        i = 0
        while i < len(events):
            event = events[i]
            if event.is_renamed:
                if i + 1 < len(events) and events[i + 1].is_renamed and (events[i + 1].event_id == event.event_id + 1):
                    cls = DirMovedEvent if event.is_directory else FileMovedEvent
                    self.queue_event(cls(event.path, events[i + 1].path))
                    self.queue_event(DirModifiedEvent(os.path.dirname(event.path)))
                    self.queue_event(DirModifiedEvent(os.path.dirname(events[i + 1].path)))
                    i += 1
                elif os.path.exists(event.path):
                    cls = DirCreatedEvent if event.is_directory else FileCreatedEvent
                    self.queue_event(cls(event.path))
                    self.queue_event(DirModifiedEvent(os.path.dirname(event.path)))
                else:
                    cls = DirDeletedEvent if event.is_directory else FileDeletedEvent
                    self.queue_event(cls(event.path))
                    self.queue_event(DirModifiedEvent(os.path.dirname(event.path)))
            elif event.is_modified or event.is_inode_meta_mod or event.is_xattr_mod:
                cls = DirModifiedEvent if event.is_directory else FileModifiedEvent
                self.queue_event(cls(event.path))
            elif event.is_created:
                cls = DirCreatedEvent if event.is_directory else FileCreatedEvent
                self.queue_event(cls(event.path))
                self.queue_event(DirModifiedEvent(os.path.dirname(event.path)))
            elif event.is_removed:
                cls = DirDeletedEvent if event.is_directory else FileDeletedEvent
                self.queue_event(cls(event.path))
                self.queue_event(DirModifiedEvent(os.path.dirname(event.path)))
            i += 1