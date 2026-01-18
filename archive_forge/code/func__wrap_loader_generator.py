import collections
import os
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def _wrap_loader_generator(self, loader_generator):
    """Wraps `DirectoryLoader` generator to swallow
        `DirectoryDeletedError`."""
    try:
        for item in loader_generator:
            yield item
    except directory_watcher.DirectoryDeletedError:
        return