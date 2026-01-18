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
def _spawn_upload_async(self, event: RequestUpload, async_executor: AsyncExecutor) -> None:
    """Equivalent to _spawn_upload_sync, but uses the async event loop instead of a thread, and requires `event.save_fn_async`.

        Raises:
            AssertionError: if `event.save_fn_async` is None.
        """
    assert event.save_fn_async is not None
    self._running_jobs[event.save_name] = event

    async def run_and_notify() -> None:
        try:
            await self._do_upload_async(event)
        finally:
            self._event_queue.put(EventJobDone(event, exc=sys.exc_info()[1]))
    async_executor.submit(run_and_notify())