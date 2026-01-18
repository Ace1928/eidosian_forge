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
class StepUpload:

    def __init__(self, api: 'internal_api.Api', stats: 'stats.Stats', event_queue: 'queue.Queue[Event]', max_threads: int, file_stream: 'file_stream.FileStreamApi', settings: Optional['SettingsStatic']=None) -> None:
        self._api = api
        self._stats = stats
        self._event_queue = event_queue
        self._file_stream = file_stream
        self._thread = threading.Thread(target=self._thread_body)
        self._thread.daemon = True
        self._pool = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='wandb-upload', max_workers=max_threads)
        self._async_executor = AsyncExecutor(pool=self._pool, concurrency_limit=settings._async_upload_concurrency_limit) if settings is not None and settings._async_upload_concurrency_limit else None
        self._running_jobs: MutableMapping[LogicalPath, RequestUpload] = {}
        self._pending_jobs: MutableSequence[RequestUpload] = []
        self._artifacts: MutableMapping[str, ArtifactStatus] = {}
        self.silent = bool(settings.silent) if settings else False

    def _thread_body(self) -> None:
        event: Optional[Event]
        finish_callback = None
        while True:
            event = self._event_queue.get()
            if isinstance(event, RequestFinish):
                finish_callback = event.callback
                break
            self._handle_event(event)
        while True:
            try:
                event = self._event_queue.get(True, 0.2)
            except queue.Empty:
                event = None
            if event:
                self._handle_event(event)
            elif not self._running_jobs:
                self._pool.shutdown(wait=False)
                if self._async_executor:
                    self._async_executor.stop()
                if finish_callback:
                    finish_callback()
                break

    def _handle_event(self, event: Event) -> None:
        if isinstance(event, EventJobDone):
            job = event.job
            if event.exc is not None:
                logger.exception('Failed to upload file: %s', job.path, exc_info=event.exc)
            if job.artifact_id:
                if event.exc is None:
                    self._artifacts[job.artifact_id]['pending_count'] -= 1
                    self._maybe_commit_artifact(job.artifact_id)
                else:
                    if not self.silent:
                        termerror("Uploading artifact file failed. Artifact won't be committed.")
                    self._fail_artifact_futures(job.artifact_id, event.exc)
            self._running_jobs.pop(job.save_name)
            if self._pending_jobs:
                event = self._pending_jobs.pop(0)
                self._start_upload_job(event)
        elif isinstance(event, RequestCommitArtifact):
            if event.artifact_id not in self._artifacts:
                self._init_artifact(event.artifact_id)
            self._artifacts[event.artifact_id]['commit_requested'] = True
            self._artifacts[event.artifact_id]['finalize'] = event.finalize
            self._artifacts[event.artifact_id]['pre_commit_callbacks'].add(event.before_commit)
            self._artifacts[event.artifact_id]['result_futures'].add(event.result_future)
            self._maybe_commit_artifact(event.artifact_id)
        elif isinstance(event, RequestUpload):
            if event.artifact_id is not None:
                if event.artifact_id not in self._artifacts:
                    self._init_artifact(event.artifact_id)
                self._artifacts[event.artifact_id]['pending_count'] += 1
            self._start_upload_job(event)
        else:
            raise Exception('Programming error: unhandled event: %s' % str(event))

    def _start_upload_job(self, event: RequestUpload) -> None:
        if event.save_name in self._running_jobs:
            self._pending_jobs.append(event)
            return
        if self._async_executor and event.save_fn_async is not None:
            self._spawn_upload_async(event, async_executor=self._async_executor)
        else:
            self._spawn_upload_sync(event)

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

    def _do_upload_sync(self, event: RequestUpload) -> None:
        job = upload_job.UploadJob(self._stats, self._api, self._file_stream, self.silent, event.save_name, event.path, event.artifact_id, event.md5, event.copied, event.save_fn, event.digest)
        job.run()

    async def _do_upload_async(self, event: RequestUpload) -> None:
        """Upload a file and returns when it's done. Requires `event.save_fn_async`."""
        assert event.save_fn_async is not None
        job = upload_job.UploadJobAsync(stats=self._stats, api=self._api, file_stream=self._file_stream, silent=self.silent, request=event, save_fn_async=event.save_fn_async)
        await job.run()

    def _init_artifact(self, artifact_id: str) -> None:
        self._artifacts[artifact_id] = {'finalize': False, 'pending_count': 0, 'commit_requested': False, 'pre_commit_callbacks': set(), 'result_futures': set()}

    def _maybe_commit_artifact(self, artifact_id: str) -> None:
        artifact_status = self._artifacts[artifact_id]
        if artifact_status['pending_count'] == 0 and artifact_status['commit_requested']:
            try:
                for pre_callback in artifact_status['pre_commit_callbacks']:
                    pre_callback()
                if artifact_status['finalize']:
                    self._api.commit_artifact(artifact_id)
            except Exception as exc:
                termerror(f"Committing artifact failed. Artifact {artifact_id} won't be finalized.")
                termerror(str(exc))
                self._fail_artifact_futures(artifact_id, exc)
            else:
                self._resolve_artifact_futures(artifact_id)

    def _fail_artifact_futures(self, artifact_id: str, exc: BaseException) -> None:
        futures = self._artifacts[artifact_id]['result_futures']
        for result_future in futures:
            result_future.set_exception(exc)
        futures.clear()

    def _resolve_artifact_futures(self, artifact_id: str) -> None:
        futures = self._artifacts[artifact_id]['result_futures']
        for result_future in futures:
            result_future.set_result(None)
        futures.clear()

    def start(self) -> None:
        self._thread.start()
        if self._async_executor:
            self._async_executor.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()