from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Final
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
class PollingPathWatcher:
    """Watches a path on disk via a polling loop."""
    _executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

    @staticmethod
    def close_all() -> None:
        """Close top-level watcher object.

        This is a no-op, and exists for interface parity with
        EventBasedPathWatcher.
        """
        _LOGGER.debug('Watcher closed')

    def __init__(self, path: str, on_changed: Callable[[str], None], *, glob_pattern: str | None=None, allow_nonexistent: bool=False) -> None:
        """Constructor.

        You do not need to retain a reference to a PollingPathWatcher to
        prevent it from being garbage collected. (The global _executor object
        retains references to all active instances.)
        """
        self._path = path
        self._on_changed = on_changed
        self._glob_pattern = glob_pattern
        self._allow_nonexistent = allow_nonexistent
        self._active = True
        self._modification_time = util.path_modification_time(self._path, self._allow_nonexistent)
        self._md5 = util.calc_md5_with_blocking_retries(self._path, glob_pattern=self._glob_pattern, allow_nonexistent=self._allow_nonexistent)
        self._schedule()

    def __repr__(self) -> str:
        return repr_(self)

    def _schedule(self) -> None:

        def task():
            time.sleep(_POLLING_PERIOD_SECS)
            self._check_if_path_changed()
        PollingPathWatcher._executor.submit(task)

    def _check_if_path_changed(self) -> None:
        if not self._active:
            return
        modification_time = util.path_modification_time(self._path, self._allow_nonexistent)
        if modification_time != 0.0 and modification_time <= self._modification_time:
            self._schedule()
            return
        self._modification_time = modification_time
        md5 = util.calc_md5_with_blocking_retries(self._path, glob_pattern=self._glob_pattern, allow_nonexistent=self._allow_nonexistent)
        if md5 == self._md5:
            self._schedule()
            return
        self._md5 = md5
        _LOGGER.debug('Change detected: %s', self._path)
        self._on_changed(self._path)
        self._schedule()

    def close(self) -> None:
        """Stop watching the file system."""
        self._active = False