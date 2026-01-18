from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
class BaseObserver(EventDispatcher):
    """Base observer."""

    def __init__(self, emitter_class, timeout=DEFAULT_OBSERVER_TIMEOUT):
        EventDispatcher.__init__(self, timeout)
        self._emitter_class = emitter_class
        self._lock = threading.RLock()
        self._watches = set()
        self._handlers = dict()
        self._emitters = set()
        self._emitter_for_watch = dict()

    def _add_emitter(self, emitter):
        self._emitter_for_watch[emitter.watch] = emitter
        self._emitters.add(emitter)

    def _remove_emitter(self, emitter):
        del self._emitter_for_watch[emitter.watch]
        self._emitters.remove(emitter)
        emitter.stop()
        try:
            emitter.join()
        except RuntimeError:
            pass

    def _clear_emitters(self):
        for emitter in self._emitters:
            emitter.stop()
        for emitter in self._emitters:
            try:
                emitter.join()
            except RuntimeError:
                pass
        self._emitters.clear()
        self._emitter_for_watch.clear()

    def _add_handler_for_watch(self, event_handler, watch):
        if watch not in self._handlers:
            self._handlers[watch] = set()
        self._handlers[watch].add(event_handler)

    def _remove_handlers_for_watch(self, watch):
        del self._handlers[watch]

    @property
    def emitters(self):
        """Returns event emitter created by this observer."""
        return self._emitters

    def start(self):
        for emitter in self._emitters:
            emitter.start()
        super(BaseObserver, self).start()

    def schedule(self, event_handler, path, recursive=False):
        """
        Schedules watching a path and calls appropriate methods specified
        in the given event handler in response to file system events.

        :param event_handler:
            An event handler instance that has appropriate event handling
            methods which will be called by the observer in response to
            file system events.
        :type event_handler:
            :class:`watchdog.events.FileSystemEventHandler` or a subclass
        :param path:
            Directory path that will be monitored.
        :type path:
            ``str``
        :param recursive:
            ``True`` if events will be emitted for sub-directories
            traversed recursively; ``False`` otherwise.
        :type recursive:
            ``bool``
        :return:
            An :class:`ObservedWatch` object instance representing
            a watch.
        """
        with self._lock:
            watch = ObservedWatch(path, recursive)
            self._add_handler_for_watch(event_handler, watch)
            if self._emitter_for_watch.get(watch) is None:
                emitter = self._emitter_class(event_queue=self.event_queue, watch=watch, timeout=self.timeout)
                self._add_emitter(emitter)
                if self.is_alive():
                    emitter.start()
            self._watches.add(watch)
        return watch

    def add_handler_for_watch(self, event_handler, watch):
        """Adds a handler for the given watch.

        :param event_handler:
            An event handler instance that has appropriate event handling
            methods which will be called by the observer in response to
            file system events.
        :type event_handler:
            :class:`watchdog.events.FileSystemEventHandler` or a subclass
        :param watch:
            The watch to add a handler for.
        :type watch:
            An instance of :class:`ObservedWatch` or a subclass of
            :class:`ObservedWatch`
        """
        with self._lock:
            self._add_handler_for_watch(event_handler, watch)

    def remove_handler_for_watch(self, event_handler, watch):
        """Removes a handler for the given watch.

        :param event_handler:
            An event handler instance that has appropriate event handling
            methods which will be called by the observer in response to
            file system events.
        :type event_handler:
            :class:`watchdog.events.FileSystemEventHandler` or a subclass
        :param watch:
            The watch to remove a handler for.
        :type watch:
            An instance of :class:`ObservedWatch` or a subclass of
            :class:`ObservedWatch`
        """
        with self._lock:
            self._handlers[watch].remove(event_handler)

    def unschedule(self, watch):
        """Unschedules a watch.

        :param watch:
            The watch to unschedule.
        :type watch:
            An instance of :class:`ObservedWatch` or a subclass of
            :class:`ObservedWatch`
        """
        with self._lock:
            emitter = self._emitter_for_watch[watch]
            del self._handlers[watch]
            self._remove_emitter(emitter)
            self._watches.remove(watch)

    def unschedule_all(self):
        """Unschedules all watches and detaches all associated event
        handlers."""
        with self._lock:
            self._handlers.clear()
            self._clear_emitters()
            self._watches.clear()

    def on_thread_stop(self):
        self.unschedule_all()

    def dispatch_events(self, event_queue, timeout):
        event, watch = event_queue.get(block=True, timeout=timeout)
        with self._lock:
            for handler in list(self._handlers.get(watch, [])):
                if handler in self._handlers.get(watch, []):
                    handler.dispatch(event)
        event_queue.task_done()