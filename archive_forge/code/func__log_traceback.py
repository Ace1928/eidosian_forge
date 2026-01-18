import concurrent.futures
import contextvars
import logging
import sys
from types import GenericAlias
from . import base_futures
from . import events
from . import exceptions
from . import format_helpers
@_log_traceback.setter
def _log_traceback(self, val):
    if val:
        raise ValueError('_log_traceback can only be set to False')
    self.__log_traceback = False