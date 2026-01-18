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
def _fail_artifact_futures(self, artifact_id: str, exc: BaseException) -> None:
    futures = self._artifacts[artifact_id]['result_futures']
    for result_future in futures:
        result_future.set_exception(exc)
    futures.clear()