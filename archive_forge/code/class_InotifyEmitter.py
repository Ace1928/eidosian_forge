from __future__ import with_statement
import os
import threading
from .inotify_buffer import InotifyBuffer
from wandb_watchdog.observers.api import (
from wandb_watchdog.events import (
from wandb_watchdog.utils import unicode_paths
class InotifyEmitter(EventEmitter):
    """
    inotify(7)-based event emitter.

    :param event_queue:
        The event queue to fill with events.
    :param watch:
        A watch object representing the directory to monitor.
    :type watch:
        :class:`watchdog.observers.api.ObservedWatch`
    :param timeout:
        Read events blocking timeout (in seconds).
    :type timeout:
        ``float``
    """

    def __init__(self, event_queue, watch, timeout=DEFAULT_EMITTER_TIMEOUT):
        EventEmitter.__init__(self, event_queue, watch, timeout)
        self._lock = threading.Lock()
        self._inotify = None

    def on_thread_start(self):
        path = unicode_paths.encode(self.watch.path)
        self._inotify = InotifyBuffer(path, self.watch.is_recursive)

    def on_thread_stop(self):
        if self._inotify:
            self._inotify.close()

    def queue_events(self, timeout, full_events=False):
        with self._lock:
            event = self._inotify.read_event()
            if event is None:
                return
            if isinstance(event, tuple):
                move_from, move_to = event
                src_path = self._decode_path(move_from.src_path)
                dest_path = self._decode_path(move_to.src_path)
                cls = DirMovedEvent if move_from.is_directory else FileMovedEvent
                self.queue_event(cls(src_path, dest_path))
                self.queue_event(DirModifiedEvent(os.path.dirname(src_path)))
                self.queue_event(DirModifiedEvent(os.path.dirname(dest_path)))
                if move_from.is_directory and self.watch.is_recursive:
                    for sub_event in generate_sub_moved_events(src_path, dest_path):
                        self.queue_event(sub_event)
                return
            src_path = self._decode_path(event.src_path)
            if event.is_moved_to:
                if full_events:
                    cls = DirMovedEvent if event.is_directory else FileMovedEvent
                    self.queue_event(cls(None, src_path))
                else:
                    cls = DirCreatedEvent if event.is_directory else FileCreatedEvent
                    self.queue_event(cls(src_path))
                self.queue_event(DirModifiedEvent(os.path.dirname(src_path)))
                if event.is_directory and self.watch.is_recursive:
                    for sub_event in generate_sub_created_events(src_path):
                        self.queue_event(sub_event)
            elif event.is_attrib:
                cls = DirModifiedEvent if event.is_directory else FileModifiedEvent
                self.queue_event(cls(src_path))
            elif event.is_modify:
                cls = DirModifiedEvent if event.is_directory else FileModifiedEvent
                self.queue_event(cls(src_path))
            elif event.is_delete or (event.is_moved_from and (not full_events)):
                cls = DirDeletedEvent if event.is_directory else FileDeletedEvent
                self.queue_event(cls(src_path))
                self.queue_event(DirModifiedEvent(os.path.dirname(src_path)))
            elif event.is_moved_from and full_events:
                cls = DirMovedEvent if event.is_directory else FileMovedEvent
                self.queue_event(cls(src_path, None))
                self.queue_event(DirModifiedEvent(os.path.dirname(src_path)))
            elif event.is_create:
                cls = DirCreatedEvent if event.is_directory else FileCreatedEvent
                self.queue_event(cls(src_path))
                self.queue_event(DirModifiedEvent(os.path.dirname(src_path)))

    def _decode_path(self, path):
        """ Decode path only if unicode string was passed to this emitter. """
        if isinstance(self.watch.path, bytes):
            return path
        return unicode_paths.decode(path)