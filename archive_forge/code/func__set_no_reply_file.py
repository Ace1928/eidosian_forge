import json
import os
import signal
import socket
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from torch.distributed.elastic.utils.logging import get_logger
from .error_handler import ErrorHandler  # noqa: F401
from .handlers import get_error_handler  # noqa: F401
def _set_no_reply_file(self):
    self.error_file = _NOT_AVAILABLE
    self.error_file_data = _EMPTY_ERROR_DATA
    self.message = ''
    self.timestamp = int(time.time())