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
def _flush(self) -> None:
    if not self._data:
        return
    if self._step_size > util.MAX_LINE_BYTES:
        metrics = [(k, sys.getsizeof(v)) for k, v in self._data.items()]
        metrics.sort(key=lambda t: t[1], reverse=True)
        bad = 0
        dropped_keys = []
        for k, v in metrics:
            if self._step_size - bad < util.MAX_LINE_BYTES - 100000:
                break
            else:
                bad += v
                dropped_keys.append(k)
                del self._data[k]
        wandb.termwarn('Step {} exceeds max data limit, dropping {} of the largest keys:'.format(self._step, len(dropped_keys)))
        print('\t' + '\n\t'.join(dropped_keys))
    self._data['_step'] = self._step
    self._added.append(self._data)
    self._step += 1
    self._step_size = 0