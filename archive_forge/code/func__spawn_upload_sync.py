import asyncio
import concurrent.futures
import logging
import queue
import sys
import threading
from typing import (
from wandb.errors.term import termerror
from wandb.filesync import upload_job
from wandb.sdk.lib.paths import LogicalPath
def _spawn_upload_sync(self, event: RequestUpload) -> None:
    """Spawn an upload job, and handles the bookkeeping of `self._running_jobs`.

        Context: it's important that, whenever we add an entry to `self._running_jobs`,
        we ensure that a corresponding `EventJobDone` message will eventually get handled;
        otherwise, the `_running_jobs` entry will never get removed, and the StepUpload
        will never shut down.

        The sole purpose of this function is to make sure that the code that adds an entry
        to `self._running_jobs` is textually right next to the code that eventually enqueues
        the `EventJobDone` message. This should help keep them in sync.
        """
    self._running_jobs[event.save_name] = event

    def run_and_notify() -> None:
        try:
            self._do_upload_sync(event)
        finally:
            self._event_queue.put(EventJobDone(event, exc=sys.exc_info()[1]))
    self._pool.submit(run_and_notify)