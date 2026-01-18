from __future__ import annotations
import os
import threading
from typing import Callable, Final, cast
from blinker import ANY, Signal
from watchdog import events
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
class EventBasedPathWatcher:
    """Watches a single path on disk using watchdog"""

    @staticmethod
    def close_all() -> None:
        """Close the _MultiPathWatcher singleton."""
        path_watcher = _MultiPathWatcher.get_singleton()
        path_watcher.close()
        _LOGGER.debug('Watcher closed')

    def __init__(self, path: str, on_changed: Callable[[str], None], *, glob_pattern: str | None=None, allow_nonexistent: bool=False) -> None:
        """Constructor for EventBasedPathWatchers.

        Parameters
        ----------
        path : str
            The path to watch.
        on_changed : Callable[[str], None]
            Callback to call when the path changes.
        glob_pattern : str or None
            A glob pattern to filter the files in a directory that should be
            watched. Only relevant when creating an EventBasedPathWatcher on a
            directory.
        allow_nonexistent : bool
            If True, the watcher will not raise an exception if the path does
            not exist. This can be used to watch for the creation of a file or
            directory at a given path.
        """
        self._path = os.path.abspath(path)
        self._on_changed = on_changed
        path_watcher = _MultiPathWatcher.get_singleton()
        path_watcher.watch_path(self._path, on_changed, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)
        _LOGGER.debug('Watcher created for %s', self._path)

    def __repr__(self) -> str:
        return repr_(self)

    def close(self) -> None:
        """Stop watching the path corresponding to this EventBasedPathWatcher."""
        path_watcher = _MultiPathWatcher.get_singleton()
        path_watcher.stop_watching_path(self._path, self._on_changed)