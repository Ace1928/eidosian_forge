import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from queue import Queue
from typing import (
import requests
import wandb
from wandb import util
from wandb.errors import CommError, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.filesync.dir_watcher import DirWatcher
from wandb.proto import wandb_internal_pb2
from wandb.sdk.artifacts.artifact_saver import ArtifactSaver
from wandb.sdk.interface import interface
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import (
from wandb.sdk.internal.file_pusher import FilePusher
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import (
from wandb.sdk.lib.mailbox import ContextCancelledError
from wandb.sdk.lib.proto_util import message_to_dict
def _start_run_threads(self, file_dir: Optional[str]=None) -> None:
    assert self._run
    self._fs = file_stream.FileStreamApi(self._api, self._run.run_id, self._run.start_time.ToMicroseconds() / 1000000.0, timeout=self._settings._file_stream_timeout_seconds, settings=self._api_settings)
    self._fs.set_file_policy('wandb-summary.json', file_stream.SummaryFilePolicy())
    self._fs.set_file_policy('wandb-history.jsonl', file_stream.JsonlFilePolicy(start_chunk_id=self._resume_state.history))
    self._fs.set_file_policy('wandb-events.jsonl', file_stream.JsonlFilePolicy(start_chunk_id=self._resume_state.events))
    self._fs.set_file_policy('output.log', file_stream.CRDedupeFilePolicy(start_chunk_id=self._resume_state.output))
    run_settings = message_to_dict(self._run)
    _settings = dict(self._settings)
    _settings.update(run_settings)
    wandb._sentry.configure_scope(tags=_settings, process_context='internal')
    self._fs.start()
    self._pusher = FilePusher(self._api, self._fs, settings=self._settings)
    self._dir_watcher = DirWatcher(self._settings, self._pusher, file_dir)
    logger.info('run started: %s with start time %s', self._run.run_id, self._run.start_time.ToMicroseconds() / 1000000.0)