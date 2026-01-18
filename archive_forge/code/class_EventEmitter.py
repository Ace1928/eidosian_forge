from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
class EventEmitter(BaseThread):
    """
    Producer thread base class subclassed by event emitters
    that generate events and populate a queue with them.

    :param event_queue:
        The event queue to populate with generated events.
    :type event_queue:
        :class:`watchdog.events.EventQueue`
    :param watch:
        The watch to observe and produce events for.
    :type watch:
        :class:`ObservedWatch`
    :param timeout:
        Timeout (in seconds) between successive attempts at reading events.
    :type timeout:
        ``float``
    """

    def __init__(self, event_queue, watch, timeout=DEFAULT_EMITTER_TIMEOUT):
        BaseThread.__init__(self)
        self._event_queue = event_queue
        self._watch = watch
        self._timeout = timeout

    @property
    def timeout(self):
        """
        Blocking timeout for reading events.
        """
        return self._timeout

    @property
    def watch(self):
        """
        The watch associated with this emitter.
        """
        return self._watch

    def queue_event(self, event):
        """
        Queues a single event.

        :param event:
            Event to be queued.
        :type event:
            An instance of :class:`watchdog.events.FileSystemEvent`
            or a subclass.
        """
        self._event_queue.put((event, self.watch))

    def queue_events(self, timeout):
        """Override this method to populate the event queue with events
        per interval period.

        :param timeout:
            Timeout (in seconds) between successive attempts at
            reading events.
        :type timeout:
            ``float``
        """

    def run(self):
        try:
            while self.should_keep_running():
                self.queue_events(self.timeout)
        finally:
            pass