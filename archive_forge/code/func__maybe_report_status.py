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
def _maybe_report_status(self, always: bool=False) -> None:
    time_now = time.monotonic()
    if not always and time_now < self._debounce_status_time + self.UPDATE_STATUS_TIME:
        return
    self._debounce_status_time = time_now
    status_report = wandb_internal_pb2.StatusReportRequest(record_num=self._send_record_num, sent_offset=self._send_end_offset)
    status_time = time.time()
    status_report.sync_time.FromMicroseconds(int(status_time * 1000000.0))
    record = self._interface._make_request(status_report=status_report)
    self._interface._publish(record)