import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _reconnect(self, db):
    if not self._check_reconnect_time():
        return db
    else:
        self.last_connect_attempt = time.time()
        return self._get_new_connection()