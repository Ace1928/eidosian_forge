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
def _update_summary(self) -> None:
    summary_dict = self._cached_summary.copy()
    summary_dict.pop('_wandb', None)
    if self._metadata_summary:
        summary_dict['_wandb'] = self._metadata_summary
    self._consolidated_summary.update(summary_dict)
    json_summary = json.dumps(self._consolidated_summary)
    if self._fs:
        self._fs.push(filenames.SUMMARY_FNAME, json_summary)
    summary_path = os.path.join(self._settings.files_dir, filenames.SUMMARY_FNAME)
    with open(summary_path, 'w') as f:
        f.write(json_summary)
    self._save_file(interface.GlobStr(filenames.SUMMARY_FNAME))