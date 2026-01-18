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
def _robust_scan(self, ds):
    """Attempt to scan data, handling incomplete files."""
    try:
        return ds.scan_data()
    except AssertionError as e:
        if ds.in_last_block():
            wandb.termwarn(f".wandb file is incomplete ({e}), be sure to sync this run again once it's finished")
            return None
        else:
            raise e