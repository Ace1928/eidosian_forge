import os
import queue
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import (
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def Worker():
    """Keeps reloading accumulators til none are left."""
    while True:
        try:
            name, accumulator = items_queue.get(block=False)
        except queue.Empty:
            break
        try:
            accumulator.Reload()
        except (OSError, IOError) as e:
            logger.error('Unable to reload accumulator %r: %s', name, e)
        except directory_watcher.DirectoryDeletedError:
            with names_to_delete_mutex:
                names_to_delete.add(name)
        finally:
            items_queue.task_done()