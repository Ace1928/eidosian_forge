import asyncio
import logging
import os
from typing import TYPE_CHECKING, Optional
import wandb
from wandb.sdk.lib.paths import LogicalPath
class UploadJobAsync:
    """Roughly an async equivalent of UploadJob.

    Important differences:
    - `run` is a coroutine
    - If `run()` fails, it falls back to the synchronous UploadJob
    """

    def __init__(self, stats: 'stats.Stats', api: 'internal_api.Api', file_stream: 'file_stream.FileStreamApi', silent: bool, request: 'step_upload.RequestUpload', save_fn_async: 'step_upload.SaveFnAsync') -> None:
        self._stats = stats
        self._api = api
        self._file_stream = file_stream
        self.silent = silent
        self._request = request
        self._save_fn_async = save_fn_async

    async def run(self) -> None:
        try:
            deduped = await self._save_fn_async(lambda _, t: self._stats.update_uploaded_file(self._request.path, t))
        except Exception as e:
            loop = asyncio.get_event_loop()
            logger.exception('async upload failed', exc_info=e)
            loop.run_in_executor(None, wandb._sentry.exception, e)
            wandb.termwarn('Async file upload failed; falling back to sync', repeat=False)
            sync_job = UploadJob(self._stats, self._api, self._file_stream, self.silent, self._request.save_name, self._request.path, self._request.artifact_id, self._request.md5, self._request.copied, self._request.save_fn, self._request.digest)
            await loop.run_in_executor(None, sync_job.run)
        else:
            self._file_stream.push_success(self._request.artifact_id, self._request.save_name)
            if deduped:
                logger.info('Skipped uploading %s', self._request.path)
                self._stats.set_file_deduped(self._request.path)
            else:
                logger.info('Uploaded file %s', self._request.path)
        finally:
            if self._request.copied:
                try:
                    os.remove(self._request.path)
                except OSError:
                    pass