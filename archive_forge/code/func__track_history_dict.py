import glob
import logging
import os
import queue
import socket
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib import filesystem
from wandb.viz import CustomChart
from . import run as internal_run
def _track_history_dict(self, d: 'HistoryDict') -> 'HistoryDict':
    e = {}
    for k in d.keys():
        e[k] = d[k]
        self._step_size += sys.getsizeof(e[k])
    return e