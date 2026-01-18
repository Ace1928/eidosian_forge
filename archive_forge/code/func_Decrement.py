from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import errno
import logging
import multiprocessing
import threading
import traceback
from gslib.utils import constants
from gslib.utils import system_util
from six.moves import queue as Queue
def Decrement(self):
    if self.multiprocessing_is_available:
        self.value.value -= 1
    else:
        with self.lock:
            self.value -= 1