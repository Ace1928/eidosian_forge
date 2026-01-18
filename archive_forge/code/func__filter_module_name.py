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
def _filter_module_name(self, name: str) -> bool:
    """
        Filter based on module name

        - True if the module is not registered and is_global is False 
        - False if the module is registered and is_global is False
        """
    _is_registered = is_registered_logger_module(name)
    if self.is_global:
        return _is_registered is not False
    return _is_registered is False