import re
import datetime
import numpy as np
import csv
import ctypes
@staticmethod
def _get_date_format(atrv):
    m = r_date.match(atrv)
    if m:
        pattern = m.group(1).strip()
        datetime_unit = None
        if 'yyyy' in pattern:
            pattern = pattern.replace('yyyy', '%Y')
            datetime_unit = 'Y'
        elif 'yy':
            pattern = pattern.replace('yy', '%y')
            datetime_unit = 'Y'
        if 'MM' in pattern:
            pattern = pattern.replace('MM', '%m')
            datetime_unit = 'M'
        if 'dd' in pattern:
            pattern = pattern.replace('dd', '%d')
            datetime_unit = 'D'
        if 'HH' in pattern:
            pattern = pattern.replace('HH', '%H')
            datetime_unit = 'h'
        if 'mm' in pattern:
            pattern = pattern.replace('mm', '%M')
            datetime_unit = 'm'
        if 'ss' in pattern:
            pattern = pattern.replace('ss', '%S')
            datetime_unit = 's'
        if 'z' in pattern or 'Z' in pattern:
            raise ValueError('Date type attributes with time zone not supported, yet')
        if datetime_unit is None:
            raise ValueError('Invalid or unsupported date format')
        return (pattern, datetime_unit)
    else:
        raise ValueError('Invalid or no date format')