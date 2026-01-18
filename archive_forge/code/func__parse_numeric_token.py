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
def _parse_numeric_token(self, tokens, idx, info, ymd, res, fuzzy):
    value_repr = tokens[idx]
    try:
        value = self._to_decimal(value_repr)
    except Exception as e:
        six.raise_from(ValueError('Unknown numeric token'), e)
    len_li = len(value_repr)
    len_l = len(tokens)
    if len(ymd) == 3 and len_li in (2, 4) and (res.hour is None) and (idx + 1 >= len_l or (tokens[idx + 1] != ':' and info.hms(tokens[idx + 1]) is None)):
        s = tokens[idx]
        res.hour = int(s[:2])
        if len_li == 4:
            res.minute = int(s[2:])
    elif len_li == 6 or (len_li > 6 and tokens[idx].find('.') == 6):
        s = tokens[idx]
        if not ymd and '.' not in tokens[idx]:
            ymd.append(s[:2])
            ymd.append(s[2:4])
            ymd.append(s[4:])
        else:
            res.hour = int(s[:2])
            res.minute = int(s[2:4])
            res.second, res.microsecond = self._parsems(s[4:])
    elif len_li in (8, 12, 14):
        s = tokens[idx]
        ymd.append(s[:4], 'Y')
        ymd.append(s[4:6])
        ymd.append(s[6:8])
        if len_li > 8:
            res.hour = int(s[8:10])
            res.minute = int(s[10:12])
            if len_li > 12:
                res.second = int(s[12:])
    elif self._find_hms_idx(idx, tokens, info, allow_jump=True) is not None:
        hms_idx = self._find_hms_idx(idx, tokens, info, allow_jump=True)
        idx, hms = self._parse_hms(idx, tokens, info, hms_idx)
        if hms is not None:
            self._assign_hms(res, value_repr, hms)
    elif idx + 2 < len_l and tokens[idx + 1] == ':':
        res.hour = int(value)
        value = self._to_decimal(tokens[idx + 2])
        res.minute, res.second = self._parse_min_sec(value)
        if idx + 4 < len_l and tokens[idx + 3] == ':':
            res.second, res.microsecond = self._parsems(tokens[idx + 4])
            idx += 2
        idx += 2
    elif idx + 1 < len_l and tokens[idx + 1] in ('-', '/', '.'):
        sep = tokens[idx + 1]
        ymd.append(value_repr)
        if idx + 2 < len_l and (not info.jump(tokens[idx + 2])):
            if tokens[idx + 2].isdigit():
                ymd.append(tokens[idx + 2])
            else:
                value = info.month(tokens[idx + 2])
                if value is not None:
                    ymd.append(value, 'M')
                else:
                    raise ValueError()
            if idx + 3 < len_l and tokens[idx + 3] == sep:
                value = info.month(tokens[idx + 4])
                if value is not None:
                    ymd.append(value, 'M')
                else:
                    ymd.append(tokens[idx + 4])
                idx += 2
            idx += 1
        idx += 1
    elif idx + 1 >= len_l or info.jump(tokens[idx + 1]):
        if idx + 2 < len_l and info.ampm(tokens[idx + 2]) is not None:
            hour = int(value)
            res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 2]))
            idx += 1
        else:
            ymd.append(value)
        idx += 1
    elif info.ampm(tokens[idx + 1]) is not None and 0 <= value < 24:
        hour = int(value)
        res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 1]))
        idx += 1
    elif ymd.could_be_day(value):
        ymd.append(value)
    elif not fuzzy:
        raise ValueError()
    return idx