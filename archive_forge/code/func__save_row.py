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
def _save_row(self, row: 'HistoryDict') -> None:
    chart_keys = set()
    for k in row:
        if isinstance(row[k], CustomChart):
            chart_keys.add(k)
            key = row[k].get_config_key(k)
            value = row[k].get_config_value('Vega2', row[k].user_query(f'{k}_table'))
            row[k] = row[k]._data
            self._tbwatcher._interface.publish_config(val=value, key=key)
    for k in chart_keys:
        row[f'{k}_table'] = row.pop(k)
    self._tbwatcher._interface.publish_history(row, run=self._internal_run, publish_step=False)