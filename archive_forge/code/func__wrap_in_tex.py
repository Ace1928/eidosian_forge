import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
def _wrap_in_tex(text):
    p = '([a-zA-Z]+)'
    ret_text = re.sub(p, '}$\\1$\\\\mathdefault{', text)
    ret_text = ret_text.replace('-', '{-}').replace(':', '{:}')
    ret_text = ret_text.replace(' ', '\\;')
    ret_text = '$\\mathdefault{' + ret_text + '}$'
    ret_text = ret_text.replace('$\\mathdefault{}$', '')
    return ret_text