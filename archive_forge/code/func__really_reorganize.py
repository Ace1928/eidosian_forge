import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
def _really_reorganize(self):
    self.log.debug('reorganizing the database')
    key = self.db.firstkey()
    breakpoint = time.time() - self.max_age
    while key is not None:
        rec = self._really_getitem(key)
        delkey = None
        if int(time.mktime(rec.r_updated.timetuple())) < breakpoint:
            self.log.debug('deleting key %s', key)
            delkey = key
        key = self.db.nextkey(key)
        if delkey:
            self._really_delitem(delkey)
    self.db.reorganize()