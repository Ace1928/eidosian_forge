import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
def _really_getitem(self, key):
    return GdbmDBHandle.decode_record(self.db[key])