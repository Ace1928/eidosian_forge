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
def _find_tfevent_files(self, sync_item):
    tb_event_files = 0
    tb_logdirs = []
    tb_root = None
    if self._sync_tensorboard:
        if os.path.isdir(sync_item):
            files = []
            for dirpath, _, _files in os.walk(sync_item):
                for f in _files:
                    if TFEVENT_SUBSTRING in f:
                        files.append(os.path.join(dirpath, f))
            for tfevent in files:
                tb_event_files += 1
                tb_dir = os.path.dirname(os.path.abspath(tfevent))
                if tb_dir not in tb_logdirs:
                    tb_logdirs.append(tb_dir)
            if len(tb_logdirs) > 0:
                tb_root = os.path.dirname(os.path.commonprefix(tb_logdirs))
        elif TFEVENT_SUBSTRING in sync_item:
            tb_root = os.path.dirname(os.path.abspath(sync_item))
            tb_logdirs.append(tb_root)
            tb_event_files = 1
    return (tb_event_files, tb_logdirs, tb_root)