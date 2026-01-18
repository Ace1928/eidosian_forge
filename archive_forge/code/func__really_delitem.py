import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
def _really_delitem(self, key):
    del self.db[key]