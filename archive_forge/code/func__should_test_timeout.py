import os
import platform
import signal
import sys
import time
import warnings
from functools import partial
from threading import Thread
from typing import List
from unittest import SkipTest, TestCase
from pytest import mark
import zmq
from zmq.utils import jsonapi
@property
def _should_test_timeout(self):
    return self._is_pyzmq_test and hasattr(signal, 'SIGALRM') and self.test_timeout_seconds