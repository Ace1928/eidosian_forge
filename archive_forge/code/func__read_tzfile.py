from relative deltas), local machine timezone, fixed offset timezone, and UTC
import datetime
import logging  # GOOGLE
import struct
import time
import sys
import os
import bisect
import weakref
from collections import OrderedDict
import six
from six import string_types
from six.moves import _thread
from ._common import tzname_in_python2, _tzinfo
from ._common import tzrangebase, enfold
from ._common import _validate_fromutc_inputs
from ._factories import _TzSingleton, _TzOffsetFactory
from ._factories import _TzStrFactory
from warnings import warn
def _read_tzfile(self, fileobj):
    out = _tzfile()
    if fileobj.read(4).decode() != 'TZif':
        raise ValueError('magic not found')
    fileobj.read(16)
    ttisgmtcnt, ttisstdcnt, leapcnt, timecnt, typecnt, charcnt = struct.unpack('>6l', fileobj.read(24))
    if timecnt:
        out.trans_list_utc = list(struct.unpack('>%dl' % timecnt, fileobj.read(timecnt * 4)))
    else:
        out.trans_list_utc = []
    if timecnt:
        out.trans_idx = struct.unpack('>%dB' % timecnt, fileobj.read(timecnt))
    else:
        out.trans_idx = []
    ttinfo = []
    for i in range(typecnt):
        ttinfo.append(struct.unpack('>lbb', fileobj.read(6)))
    abbr = fileobj.read(charcnt).decode()
    if leapcnt:
        fileobj.seek(leapcnt * 8, os.SEEK_CUR)
    if ttisstdcnt:
        isstd = struct.unpack('>%db' % ttisstdcnt, fileobj.read(ttisstdcnt))
    if ttisgmtcnt:
        isgmt = struct.unpack('>%db' % ttisgmtcnt, fileobj.read(ttisgmtcnt))
    out.ttinfo_list = []
    for i in range(typecnt):
        gmtoff, isdst, abbrind = ttinfo[i]
        gmtoff = _get_supported_offset(gmtoff)
        tti = _ttinfo()
        tti.offset = gmtoff
        tti.dstoffset = datetime.timedelta(0)
        tti.delta = datetime.timedelta(seconds=gmtoff)
        tti.isdst = isdst
        tti.abbr = abbr[abbrind:abbr.find('\x00', abbrind)]
        tti.isstd = ttisstdcnt > i and isstd[i] != 0
        tti.isgmt = ttisgmtcnt > i and isgmt[i] != 0
        out.ttinfo_list.append(tti)
    out.trans_idx = [out.ttinfo_list[idx] for idx in out.trans_idx]
    out.ttinfo_std = None
    out.ttinfo_dst = None
    out.ttinfo_before = None
    if out.ttinfo_list:
        if not out.trans_list_utc:
            out.ttinfo_std = out.ttinfo_first = out.ttinfo_list[0]
        else:
            for i in range(timecnt - 1, -1, -1):
                tti = out.trans_idx[i]
                if not out.ttinfo_std and (not tti.isdst):
                    out.ttinfo_std = tti
                elif not out.ttinfo_dst and tti.isdst:
                    out.ttinfo_dst = tti
                if out.ttinfo_std and out.ttinfo_dst:
                    break
            else:
                if out.ttinfo_dst and (not out.ttinfo_std):
                    out.ttinfo_std = out.ttinfo_dst
            for tti in out.ttinfo_list:
                if not tti.isdst:
                    out.ttinfo_before = tti
                    break
            else:
                out.ttinfo_before = out.ttinfo_list[0]
    lastdst = None
    lastoffset = None
    lastdstoffset = None
    lastbaseoffset = None
    out.trans_list = []
    for i, tti in enumerate(out.trans_idx):
        offset = tti.offset
        dstoffset = 0
        if lastdst is not None:
            if tti.isdst:
                if not lastdst:
                    dstoffset = offset - lastoffset
                if not dstoffset and lastdstoffset:
                    dstoffset = lastdstoffset
                tti.dstoffset = datetime.timedelta(seconds=dstoffset)
                lastdstoffset = dstoffset
        baseoffset = offset - dstoffset
        adjustment = baseoffset
        if lastbaseoffset is not None and baseoffset != lastbaseoffset and (tti.isdst != lastdst):
            adjustment = lastbaseoffset
        lastdst = tti.isdst
        lastoffset = offset
        lastbaseoffset = baseoffset
        out.trans_list.append(out.trans_list_utc[i] + adjustment)
    out.trans_idx = tuple(out.trans_idx)
    out.trans_list = tuple(out.trans_list)
    out.trans_list_utc = tuple(out.trans_list_utc)
    return out