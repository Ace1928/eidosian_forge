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
def _do_upload_sync(self, event: RequestUpload) -> None:
    job = upload_job.UploadJob(self._stats, self._api, self._file_stream, self.silent, event.save_name, event.path, event.artifact_id, event.md5, event.copied, event.save_fn, event.digest)
    job.run()