import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _get_connection(self):
    if self.bound:
        return self.db_queue.get()
    else:
        return self._get_new_connection()