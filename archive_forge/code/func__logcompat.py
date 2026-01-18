from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
def _logcompat(self, level, from_decorator, options, message, args, kwargs):
    """
        Compatible to < 0.6.0
        """
    try:
        self._log(level, from_decorator, options, message, args, kwargs)
    except TypeError:
        static_log_no = REVERSE_LOGLEVEL_MAPPING.get(level, 20)
        self._log(level, static_log_no, from_decorator, options, message, args, kwargs)