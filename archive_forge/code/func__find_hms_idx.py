from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
def _find_hms_idx(self, idx, tokens, info, allow_jump):
    len_l = len(tokens)
    if idx + 1 < len_l and info.hms(tokens[idx + 1]) is not None:
        hms_idx = idx + 1
    elif allow_jump and idx + 2 < len_l and (tokens[idx + 1] == ' ') and (info.hms(tokens[idx + 2]) is not None):
        hms_idx = idx + 2
    elif idx > 0 and info.hms(tokens[idx - 1]) is not None:
        hms_idx = idx - 1
    elif 1 < idx == len_l - 1 and tokens[idx - 1] == ' ' and (info.hms(tokens[idx - 2]) is not None):
        hms_idx = idx - 2
    else:
        hms_idx = None
    return hms_idx