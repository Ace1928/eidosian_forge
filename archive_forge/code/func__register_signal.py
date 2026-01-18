import logging
import os
import re
import signal
import sys
import threading
from subprocess import call
from types import FrameType
from typing import Any, Callable, Dict, List, Set, Union
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS, _PYTHON_GREATER_EQUAL_3_8_0
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_info
@staticmethod
def _register_signal(signum: _SIGNUM, handlers: _HANDLER) -> None:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signum, handlers)