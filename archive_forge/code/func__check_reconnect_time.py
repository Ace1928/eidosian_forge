import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _check_reconnect_time(self):
    if time.time() - self.last_connect_attempt < self.reconnect_period:
        self.log.debug("Can't reconnect until %s", time.ctime(self.last_connect_attempt + self.reconnect_period))
        return False
    return True