import abc
import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntFlag
from multiprocessing import synchronize
from types import FrameType
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record
from torch.distributed.elastic.multiprocessing.redirects import (
from torch.distributed.elastic.multiprocessing.tail_log import TailLog
def _get_kill_signal() -> signal.Signals:
    """Get the kill signal. SIGKILL for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT
    else:
        return signal.SIGKILL