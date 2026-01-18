import time
import logging
import datetime
import functools
from pyzor.engines.common import *
@staticmethod
def _decode_record(r):
    if not r:
        return Record()
    return Record(r_count=int(r.get(b'r_count', 0)), r_entered=decode_date(r.get(b'r_entered', 0)), r_updated=decode_date(r.get(b'r_updated', 0)), wl_count=int(r.get(b'wl_count', 0)), wl_entered=decode_date(r.get(b'wl_entered', 0)), wl_updated=decode_date(r.get(b'wl_updated', 0)))