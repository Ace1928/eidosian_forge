import datetime
import decimal
import glob
import numbers
import os
import shutil
import string
from functools import partial
from stat import ST_DEV, ST_INO
from . import _string_parsers as string_parsers
from ._ctime_functions import get_ctime, set_ctime
from ._datetime import aware_now
class RotationTime:

    def __init__(self, step_forward, time_init=None):
        self._step_forward = step_forward
        self._time_init = time_init
        self._limit = None

    def __call__(self, message, file):
        record_time = message.record['time']
        if self._limit is None:
            filepath = os.path.realpath(file.name)
            creation_time = get_ctime(filepath)
            set_ctime(filepath, creation_time)
            start_time = datetime.datetime.fromtimestamp(creation_time, tz=datetime.timezone.utc)
            time_init = self._time_init
            if time_init is None:
                limit = start_time.astimezone(record_time.tzinfo).replace(tzinfo=None)
                limit = self._step_forward(limit)
            else:
                tzinfo = record_time.tzinfo if time_init.tzinfo is None else time_init.tzinfo
                limit = start_time.astimezone(tzinfo).replace(hour=time_init.hour, minute=time_init.minute, second=time_init.second, microsecond=time_init.microsecond)
                if limit <= start_time:
                    limit = self._step_forward(limit)
                if time_init.tzinfo is None:
                    limit = limit.replace(tzinfo=None)
            self._limit = limit
        if self._limit.tzinfo is None:
            record_time = record_time.replace(tzinfo=None)
        if record_time >= self._limit:
            while self._limit <= record_time:
                self._limit = self._step_forward(self._limit)
            return True
        return False