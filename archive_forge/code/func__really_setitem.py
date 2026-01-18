import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
def _really_setitem(self, key, value):
    self.db[key] = GdbmDBHandle.encode_record(value)