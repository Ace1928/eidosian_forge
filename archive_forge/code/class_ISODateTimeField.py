import peewee as _peewee
from threading import Lock
import os as _os
import appdirs as _ad
import atexit as _atexit
import datetime as _datetime
import pickle as _pkl
from .utils import get_yf_logger
class ISODateTimeField(_peewee.DateTimeField):

    def db_value(self, value):
        if value and isinstance(value, _datetime.datetime):
            return value.isoformat()
        return super().db_value(value)

    def python_value(self, value):
        if value and isinstance(value, str) and ('T' in value):
            return _datetime.datetime.fromisoformat(value)
        return super().python_value(value)