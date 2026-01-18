from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
from ..packages import six
def _sleep_backoff(self):
    backoff = self.get_backoff_time()
    if backoff <= 0:
        return
    time.sleep(backoff)