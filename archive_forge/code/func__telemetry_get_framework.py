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
def _telemetry_get_framework(self) -> str:
    """Get telemetry data for internal config structure."""
    imports: telemetry.TelemetryImports
    if self._telemetry_obj.HasField('imports_finish'):
        imports = self._telemetry_obj.imports_finish
    elif self._telemetry_obj.HasField('imports_init'):
        imports = self._telemetry_obj.imports_init
    else:
        return ''
    framework = next((n for f, n in _framework_priority() if getattr(imports, f, False)), '')
    return framework