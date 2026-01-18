import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _load_file(self, fobj):
    trans_idx, trans_utc, utcoff, isdst, abbr, tz_str = _common.load_data(fobj)
    dstoff = self._utcoff_to_dstoff(trans_idx, utcoff, isdst)
    trans_local = self._ts_to_local(trans_idx, trans_utc, utcoff)
    _ttinfo_list = [_ttinfo(_load_timedelta(utcoffset), _load_timedelta(dstoffset), tzname) for utcoffset, dstoffset, tzname in zip(utcoff, dstoff, abbr)]
    self._trans_utc = trans_utc
    self._trans_local = trans_local
    self._ttinfos = [_ttinfo_list[idx] for idx in trans_idx]
    for i in range(len(isdst)):
        if not isdst[i]:
            self._tti_before = _ttinfo_list[i]
            break
    else:
        if self._ttinfos:
            self._tti_before = self._ttinfos[0]
        else:
            self._tti_before = None
    if tz_str is not None and tz_str != b'':
        self._tz_after = _parse_tz_str(tz_str.decode())
    else:
        if not self._ttinfos and (not _ttinfo_list):
            raise ValueError('No time zone information found.')
        if self._ttinfos:
            self._tz_after = self._ttinfos[-1]
        else:
            self._tz_after = _ttinfo_list[-1]
    if len(_ttinfo_list) > 1 or not isinstance(self._tz_after, _ttinfo):
        self._fixed_offset = False
    elif not _ttinfo_list:
        self._fixed_offset = True
    else:
        self._fixed_offset = _ttinfo_list[0] == self._tz_after