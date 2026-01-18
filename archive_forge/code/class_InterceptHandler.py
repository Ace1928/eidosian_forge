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
class InterceptHandler(logging.Handler):
    loglevel_mapping = LOGLEVEL_MAPPING

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = self.loglevel_mapping.get(record.levelno, 'DEBUG')
        frame, depth = (logging.currentframe(), 2)
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        log = logger.bind(request_id=None)
        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())