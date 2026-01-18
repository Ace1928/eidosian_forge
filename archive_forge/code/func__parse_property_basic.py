from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def _parse_property_basic(self, s, offset, property_id, property_type, convert_time, no_conversion):
    value = None
    size = 0
    if property_type == VT_I2:
        value = i16(s, offset)
        if value >= 32768:
            value = value - 65536
        size = 2
    elif property_type == VT_UI2:
        value = i16(s, offset)
        size = 2
    elif property_type in (VT_I4, VT_INT, VT_ERROR):
        value = i32(s, offset)
        size = 4
    elif property_type in (VT_UI4, VT_UINT):
        value = i32(s, offset)
        size = 4
    elif property_type in (VT_BSTR, VT_LPSTR):
        count = i32(s, offset)
        value = s[offset + 4:offset + 4 + count - 1]
        value = value.replace(b'\x00', b'')
        size = 4 + count
    elif property_type == VT_BLOB:
        count = i32(s, offset)
        value = s[offset + 4:offset + 4 + count]
        size = 4 + count
    elif property_type == VT_LPWSTR:
        count = i32(s, offset + 4)
        value = self._decode_utf16_str(s[offset + 4:offset + 4 + count * 2])
        size = 4 + count * 2
    elif property_type == VT_FILETIME:
        value = long(i32(s, offset)) + (long(i32(s, offset + 4)) << 32)
        if convert_time and property_id not in no_conversion:
            log.debug('Converting property #%d to python datetime, value=%d=%fs' % (property_id, value, float(value) / 10000000))
            _FILETIME_null_date = datetime.datetime(1601, 1, 1, 0, 0, 0)
            log.debug('timedelta days=%d' % (value // (10 * 1000000 * 3600 * 24)))
            value = _FILETIME_null_date + datetime.timedelta(microseconds=value // 10)
        else:
            value = value // 10000000
        size = 8
    elif property_type == VT_UI1:
        value = i8(s[offset])
        size = 1
    elif property_type == VT_CLSID:
        value = _clsid(s[offset:offset + 16])
        size = 16
    elif property_type == VT_CF:
        count = i32(s, offset)
        value = s[offset + 4:offset + 4 + count]
        size = 4 + count
    elif property_type == VT_BOOL:
        value = bool(i16(s, offset))
        size = 2
    else:
        value = None
        log.debug('property id=%d: type=%d not implemented in parser yet' % (property_id, property_type))
    return (value, size)