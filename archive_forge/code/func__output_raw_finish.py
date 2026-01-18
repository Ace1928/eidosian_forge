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
def _output_raw_finish(self) -> None:
    for stream, output_raw in self._output_raw_streams.items():
        output_raw._stopped.set()
        output_raw._writer_thr.join(timeout=5)
        if output_raw._writer_thr.is_alive():
            logger.info('processing output...')
            output_raw._writer_thr.join()
        output_raw._reader_thr.join()
        self._output_raw_flush(stream)
    self._output_raw_streams = {}
    if self._output_raw_file:
        self._output_raw_file.close()
        self._output_raw_file = None