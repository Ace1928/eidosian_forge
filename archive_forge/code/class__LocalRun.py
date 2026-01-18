import atexit
import datetime
import fnmatch
import os
import queue
import sys
import tempfile
import threading
import time
from typing import List, Optional
from urllib.parse import quote as url_quote
import wandb
from wandb.proto import wandb_internal_pb2  # type: ignore
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context, datastore, handler, sender, tb_watcher
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import filesystem
from wandb.util import check_and_warn_old
class _LocalRun:

    def __init__(self, path, synced=None):
        self.path = path
        self.synced = synced
        self.offline = os.path.basename(path).startswith('offline-')
        self.datetime = datetime.datetime.strptime(os.path.basename(path).split('run-')[1].split('-')[0], '%Y%m%d_%H%M%S')

    def __str__(self):
        return self.path