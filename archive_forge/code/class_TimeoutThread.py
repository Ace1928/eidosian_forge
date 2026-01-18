import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
class TimeoutThread(threading.Thread):
    """Creates a Thread which runs (sleeps) for a time duration equal to
    waiting_time and raises a custom ServerlessTimeout exception.
    """

    def __init__(self, waiting_time, configured_timeout):
        threading.Thread.__init__(self)
        self.waiting_time = waiting_time
        self.configured_timeout = configured_timeout
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        self._stop_event.wait(self.waiting_time)
        if self._stop_event.is_set():
            return
        integer_configured_timeout = int(self.configured_timeout)
        if integer_configured_timeout < self.configured_timeout:
            integer_configured_timeout = integer_configured_timeout + 1
        raise ServerlessTimeoutWarning('WARNING : Function is expected to get timed out. Configured timeout duration = {} seconds.'.format(integer_configured_timeout))