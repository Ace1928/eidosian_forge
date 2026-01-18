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
def _read_queue(self) -> List:
    return util.read_many_from_queue(self._queue, self.MAX_ITEMS_PER_PUSH, self.rate_limit_seconds())