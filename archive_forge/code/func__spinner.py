import collections
import functools
import threading
import time
from taskflow import test
from taskflow.utils import threading_utils as tu
def _spinner(death):
    while not death.is_set():
        time.sleep(0.1)