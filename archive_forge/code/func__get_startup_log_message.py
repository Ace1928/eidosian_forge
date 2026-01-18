import os
import asyncio
import signal
import typing
import contextlib
import functools
from croniter import croniter
from aiokeydb.v2.exceptions import ConnectionError
from aiokeydb.v2.configs import settings as default_settings
from aiokeydb.v2.types.jobs import Job, CronJob, JobStatus, TaskType
from aiokeydb.v2.utils.queue import (
from aiokeydb.v2.utils.logs import logger, ColorMap
def _get_startup_log_message(self):
    """
        Builds the startup log message.
        """
    _msg = f'{self._worker_identity}: {self.worker_host}.{self.name} v{self.settings.version}'
    _msg += f'\n- {ColorMap.cyan}[Worker ID]{ColorMap.reset}: {ColorMap.bold}{self.worker_id}{ColorMap.reset}'
    if self._is_primary_worker:
        _msg += f'\n- {ColorMap.cyan}[Queue]{ColorMap.reset}: {ColorMap.bold}{self.queue_name} @ {self.queue.uri} DB: {self.queue.db_id}{ColorMap.reset}'
        _msg += f'\n- {ColorMap.cyan}[Registered]{ColorMap.reset}: {ColorMap.bold}{len(self.functions)} functions, {len(self.cron_jobs)} cron jobs{ColorMap.reset}'
        _msg += f'\n- {ColorMap.cyan}[Concurrency]{ColorMap.reset}: {ColorMap.bold}{self.concurrency}/jobs, {self.broadcast_concurrency}/broadcasts{ColorMap.reset}'
        if self.verbose_startup:
            _msg += f'\n- {ColorMap.cyan}[Serializer]{ColorMap.reset}: {self.queue.serializer}'
            _msg += f'\n- {ColorMap.cyan}[Worker Attributes]{ColorMap.reset}: {self.worker_attributes}'
            if self._is_ctx_retryable:
                _msg += f'\n- {ColorMap.cyan}[Retryable]{ColorMap.reset}: {self._is_ctx_retryable}'
            _msg += f'\n- {ColorMap.cyan}[Functions]{ColorMap.reset}:'
            for function_name in self.functions:
                _msg += f'\n   - {ColorMap.bold}{function_name}{ColorMap.reset}'
            if self.settings.worker.has_silenced_functions:
                _msg += f'\n - {ColorMap.cyan}[Silenced Functions]{ColorMap.reset}:'
                for stage, silenced_functions in self.settings.worker.silenced_function_dict.items():
                    if silenced_functions:
                        _msg += f'\n   - {stage}: {silenced_functions}'
            if self.queue.function_tracker_enabled:
                _msg += f'\n- {ColorMap.cyan}[Function Tracker Enabled]{ColorMap.reset}: {self.queue.function_tracker_enabled}'
    return _msg