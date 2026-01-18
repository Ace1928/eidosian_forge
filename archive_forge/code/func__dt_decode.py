import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
def _dt_decode(datetime_str):
    """Decode a string into a datetime object."""
    if datetime_str == 'None':
        return None
    try:
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')