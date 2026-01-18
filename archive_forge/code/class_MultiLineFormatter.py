from __future__ import print_function
import datetime as _datetime
import logging
import re as _re
import sys as _sys
import threading
from functools import lru_cache
from inspect import getmembers
from types import FunctionType
from typing import List, Optional
import numpy as _np
import pandas as _pd
import pytz as _tz
import requests as _requests
from dateutil.relativedelta import relativedelta
from pytz import UnknownTimeZoneError
from yfinance import const
from .const import _BASE_URL_
class MultiLineFormatter(logging.Formatter):

    def __init__(self, fmt):
        super().__init__(fmt)
        match = _re.search('%\\(levelname\\)-(\\d+)s', fmt)
        self.level_length = int(match.group(1)) if match else 0

    def format(self, record):
        original = super().format(record)
        lines = original.split('\n')
        levelname = lines[0].split(' ')[0]
        if len(lines) <= 1:
            return original
        else:
            formatted = [lines[0]]
            if self.level_length == 0:
                padding = ' ' * len(levelname)
            else:
                padding = ' ' * self.level_length
            padding += ' '
            formatted.extend((padding + line for line in lines[1:]))
            return '\n'.join(formatted)