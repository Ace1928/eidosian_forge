import logging
import random
import sys
import time
import traceback
from google.cloud.ml.util import _exceptions
from six import reraise
class Clock(object):
    """A simple clock implementing sleep()."""

    def sleep(self, value):
        time.sleep(value)