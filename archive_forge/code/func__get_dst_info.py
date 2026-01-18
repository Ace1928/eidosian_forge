import logging
from datetime import datetime
from tzlocal import utils
from tzlocal.windows_tz import win_tz
def _get_dst_info(tz):
    dst_offset = std_offset = None
    has_dst = False
    year = datetime.now().year
    for dt in (datetime(year, 1, 1), datetime(year, 6, 1)):
        if tz.dst(dt).total_seconds() == 0.0:
            std_offset = tz.utcoffset(dt).total_seconds()
        else:
            has_dst = True
    return (has_dst, std_offset, dst_offset)