from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def _clsid(clsid):
    """
    Converts a CLSID to a human-readable string.

    :param clsid: string of length 16.
    """
    assert len(clsid) == 16
    if not clsid.strip(b'\x00'):
        return ''
    return ('%08X-%04X-%04X-%02X%02X-' + '%02X' * 6) % ((i32(clsid, 0), i16(clsid, 4), i16(clsid, 6)) + tuple(map(i8, clsid[8:16])))