import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
def _thread_except_body(self) -> None:
    try:
        self._thread_body()
    except Exception as e:
        exc_info = sys.exc_info()
        self._exc_info = exc_info
        logger.exception('generic exception in filestream thread')
        wandb._sentry.exception(exc_info)
        raise e