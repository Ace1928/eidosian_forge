import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _parse_tz_str(tz_str):
    offset_str, *start_end_str = tz_str.split(',', 1)
    parser_re = re.compile('\n        (?P<std>[^<0-9:.+-]+|<[a-zA-Z0-9+-]+>)\n        (?:\n            (?P<stdoff>[+-]?\\d{1,3}(?::\\d{2}(?::\\d{2})?)?)\n            (?:\n                (?P<dst>[^0-9:.+-]+|<[a-zA-Z0-9+-]+>)\n                (?P<dstoff>[+-]?\\d{1,3}(?::\\d{2}(?::\\d{2})?)?)?\n            )? # dst\n        )? # stdoff\n        ', re.ASCII | re.VERBOSE)
    m = parser_re.fullmatch(offset_str)
    if m is None:
        raise ValueError(f'{tz_str} is not a valid TZ string')
    std_abbr = m.group('std')
    dst_abbr = m.group('dst')
    dst_offset = None
    std_abbr = std_abbr.strip('<>')
    if dst_abbr:
        dst_abbr = dst_abbr.strip('<>')
    if (std_offset := m.group('stdoff')):
        try:
            std_offset = _parse_tz_delta(std_offset)
        except ValueError as e:
            raise ValueError(f'Invalid STD offset in {tz_str}') from e
    else:
        std_offset = 0
    if dst_abbr is not None:
        if (dst_offset := m.group('dstoff')):
            try:
                dst_offset = _parse_tz_delta(dst_offset)
            except ValueError as e:
                raise ValueError(f'Invalid DST offset in {tz_str}') from e
        else:
            dst_offset = std_offset + 3600
        if not start_end_str:
            raise ValueError(f'Missing transition rules: {tz_str}')
        start_end_strs = start_end_str[0].split(',', 1)
        try:
            start, end = (_parse_dst_start_end(x) for x in start_end_strs)
        except ValueError as e:
            raise ValueError(f'Invalid TZ string: {tz_str}') from e
        return _TZStr(std_abbr, std_offset, dst_abbr, dst_offset, start, end)
    elif start_end_str:
        raise ValueError(f'Transition rule present without DST: {tz_str}')
    else:
        return _ttinfo(_load_timedelta(std_offset), _load_timedelta(0), std_abbr)