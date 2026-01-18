import asyncio
import atexit
import base64
import collections
import datetime
import re
import signal
import typing as t
from contextlib import asynccontextmanager, contextmanager
from queue import Empty
from textwrap import dedent
from time import monotonic
from jupyter_client import KernelManager
from jupyter_client.client import KernelClient
from nbformat import NotebookNode
from nbformat.v4 import output_from_msg
from traitlets import Any, Bool, Callable, Dict, Enum, Integer, List, Type, Unicode, default
from traitlets.config.configurable import LoggingConfigurable
from .exceptions import (
from .output_widget import OutputWidget
from .util import ensure_async, run_hook, run_sync
def _update_display_id(self, display_id: str, msg: t.Dict) -> None:
    """Update outputs with a given display_id"""
    if display_id not in self._display_id_map:
        self.log.debug('display id %r not in %s', display_id, self._display_id_map)
        return
    if msg['header']['msg_type'] == 'update_display_data':
        msg['header']['msg_type'] = 'display_data'
    try:
        out = output_from_msg(msg)
    except ValueError:
        self.log.error(f'unhandled iopub msg: {msg['msg_type']}')
        return
    for cell_idx, output_indices in self._display_id_map[display_id].items():
        cell = self.nb['cells'][cell_idx]
        outputs = cell['outputs']
        for output_idx in output_indices:
            outputs[output_idx]['data'] = out['data']
            outputs[output_idx]['metadata'] = out['metadata']