from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
def _watch_path(path: str, on_path_changed: Callable[[str], None], watcher_type: str | None=None, *, glob_pattern: str | None=None, allow_nonexistent: bool=False) -> bool:
    """Create a PathWatcher for the given path if we have a viable
    PathWatcher class.

    Parameters
    ----------
    path
        Path to watch.
    on_path_changed
        Function that's called when the path changes.
    watcher_type
        Optional watcher_type string. If None, it will default to the
        'server.fileWatcherType` config option.
    glob_pattern
        Optional glob pattern to use when watching a directory. If set, only
        files matching the pattern will be counted as being created/deleted
        within the watched directory.
    allow_nonexistent
        If True, allow the file or directory at the given path to be
        nonexistent.

    Returns
    -------
    bool
        True if the path is being watched, or False if we have no
        PathWatcher class.
    """
    if watcher_type is None:
        watcher_type = config.get_option('server.fileWatcherType')
    watcher_class = get_path_watcher_class(watcher_type)
    if watcher_class is NoOpPathWatcher:
        return False
    watcher_class(path, on_path_changed, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)
    return True