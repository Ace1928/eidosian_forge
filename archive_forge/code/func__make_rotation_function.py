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
@staticmethod
def _make_rotation_function(rotation):
    if rotation is None:
        return None
    elif isinstance(rotation, str):
        size = string_parsers.parse_size(rotation)
        if size is not None:
            return FileSink._make_rotation_function(size)
        interval = string_parsers.parse_duration(rotation)
        if interval is not None:
            return FileSink._make_rotation_function(interval)
        frequency = string_parsers.parse_frequency(rotation)
        if frequency is not None:
            return Rotation.RotationTime(frequency)
        daytime = string_parsers.parse_daytime(rotation)
        if daytime is not None:
            day, time = daytime
            if day is None:
                return FileSink._make_rotation_function(time)
            if time is None:
                time = datetime.time(0, 0, 0)
            step_forward = partial(Rotation.forward_weekday, weekday=day)
            return Rotation.RotationTime(step_forward, time)
        raise ValueError("Cannot parse rotation from: '%s'" % rotation)
    elif isinstance(rotation, (numbers.Real, decimal.Decimal)):
        return partial(Rotation.rotation_size, size_limit=rotation)
    elif isinstance(rotation, datetime.time):
        return Rotation.RotationTime(Rotation.forward_day, rotation)
    elif isinstance(rotation, datetime.timedelta):
        step_forward = partial(Rotation.forward_interval, interval=rotation)
        return Rotation.RotationTime(step_forward)
    elif callable(rotation):
        return rotation
    else:
        raise TypeError("Cannot infer rotation for objects of type: '%s'" % type(rotation).__name__)