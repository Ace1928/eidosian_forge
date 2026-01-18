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
def _is_pyzmq_test(self):
    return self.__class__.__module__.split('.', 1)[0] == __name__.split('.', 1)[0]