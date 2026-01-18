from math import log
import os
from os import path as op
import sys
import shutil
import time
from . import appdata_dir, resource_dirs
from . import StdoutProgressIndicator, urlopen
def _sizeof_fmt(num):
    """Turn number of bytes into human-readable str"""
    units = ['bytes', 'kB', 'MB', 'GB', 'TB', 'PB']
    decimals = [0, 0, 1, 2, 2, 2]
    'Human friendly file size'
    if num > 1:
        exponent = min(int(log(num, 1024)), len(units) - 1)
        quotient = float(num) / 1024 ** exponent
        unit = units[exponent]
        num_decimals = decimals[exponent]
        format_string = '{0:.%sf} {1}' % num_decimals
        return format_string.format(quotient, unit)
    return '0 bytes' if num == 0 else '1 byte'