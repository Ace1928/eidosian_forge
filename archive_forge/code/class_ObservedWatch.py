from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
class ObservedWatch(object):
    """An scheduled watch.

    :param path:
        Path string.
    :param recursive:
        ``True`` if watch is recursive; ``False`` otherwise.
    """

    def __init__(self, path, recursive):
        self._path = path
        self._is_recursive = recursive

    @property
    def path(self):
        """The path that this watch monitors."""
        return self._path

    @property
    def is_recursive(self):
        """Determines whether subdirectories are watched for the path."""
        return self._is_recursive

    @property
    def key(self):
        return (self.path, self.is_recursive)

    def __eq__(self, watch):
        return self.key == watch.key

    def __ne__(self, watch):
        return self.key != watch.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return '<ObservedWatch: path=%s, is_recursive=%s>' % (self.path, self.is_recursive)