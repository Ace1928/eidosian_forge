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
def _output_raw_writer_thread(self, stream: 'StreamLiterals') -> None:
    while True:
        output_raw = self._output_raw_streams[stream]
        if output_raw._queue.empty():
            if output_raw._stopped.is_set():
                return
            time.sleep(0.5)
            continue
        data = []
        while not output_raw._queue.empty():
            data.append(output_raw._queue.get())
        if output_raw._stopped.is_set() and sum(map(len, data)) > 100000:
            logger.warning('Terminal output too large. Logging without processing.')
            self._output_raw_flush(stream)
            for line in data:
                self._output_raw_flush(stream, line)
            return
        try:
            output_raw._emulator.write(''.join(data))
        except Exception as e:
            logger.warning(f'problem writing to output_raw emulator: {e}')