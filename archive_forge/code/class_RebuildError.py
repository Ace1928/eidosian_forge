import linecache
import sys
import time
import types
from importlib import reload
from types import ModuleType
from typing import Dict
from twisted.python import log, reflect
class RebuildError(Exception):
    """
    Exception raised when trying to rebuild a class whereas it's not possible.
    """