import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
@staticmethod
def _utcoff_to_dstoff(trans_idx, utcoffsets, isdsts):
    typecnt = len(isdsts)
    dstoffs = [0] * typecnt
    dst_cnt = sum(isdsts)
    dst_found = 0
    for i in range(1, len(trans_idx)):
        if dst_cnt == dst_found:
            break
        idx = trans_idx[i]
        dst = isdsts[idx]
        if not dst:
            continue
        if dstoffs[idx] != 0:
            continue
        dstoff = 0
        utcoff = utcoffsets[idx]
        comp_idx = trans_idx[i - 1]
        if not isdsts[comp_idx]:
            dstoff = utcoff - utcoffsets[comp_idx]
        if not dstoff and idx < typecnt - 1:
            comp_idx = trans_idx[i + 1]
            if isdsts[comp_idx]:
                continue
            dstoff = utcoff - utcoffsets[comp_idx]
        if dstoff:
            dst_found += 1
            dstoffs[idx] = dstoff
    else:
        for idx in range(typecnt):
            if not dstoffs[idx] and isdsts[idx]:
                dstoffs[idx] = 3600
    return dstoffs