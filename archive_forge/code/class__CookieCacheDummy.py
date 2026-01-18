import peewee as _peewee
from threading import Lock
import os as _os
import appdirs as _ad
import atexit as _atexit
import datetime as _datetime
import pickle as _pkl
from .utils import get_yf_logger
class _CookieCacheDummy:
    """Dummy cache to use if Cookie cache is disabled"""

    def lookup(self, tkr):
        return None

    def store(self, tkr, Cookie):
        pass

    @property
    def Cookie_db(self):
        return None