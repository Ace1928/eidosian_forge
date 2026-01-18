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
def __update_amount(self, new_amount):
    percent_done = int(round(new_amount / 100.0 * 100.0))
    all_full = self.width - 2
    num_hashes = int(round(percent_done / 100.0 * all_full))
    self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
    pct_place = len(self.prog_bar) // 2 - len(str(percent_done))
    pct_string = f'{percent_done}%%'
    self.prog_bar = self.prog_bar[0:pct_place] + (pct_string + self.prog_bar[pct_place + len(pct_string):])