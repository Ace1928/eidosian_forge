import peewee as _peewee
from threading import Lock
import os as _os
import appdirs as _ad
import atexit as _atexit
import datetime as _datetime
import pickle as _pkl
from .utils import get_yf_logger
class _CookieSchema(_peewee.Model):
    strategy = _peewee.CharField(primary_key=True)
    fetch_date = ISODateTimeField(default=_datetime.datetime.now)
    cookie_bytes = _peewee.BlobField()

    class Meta:
        database = Cookie_db_proxy
        without_rowid = True