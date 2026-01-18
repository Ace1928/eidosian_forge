import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _create_info_struct(file, mode, samplerate, channels, format, subtype, endian):
    """Check arguments and create SF_INFO struct."""
    original_format = format
    if format is None:
        format = _get_format_from_filename(file, mode)
        assert isinstance(format, (_unicode, str))
    else:
        _check_format(format)
    info = _ffi.new('SF_INFO*')
    if 'r' not in mode or format.upper() == 'RAW':
        if samplerate is None:
            raise TypeError('samplerate must be specified')
        info.samplerate = samplerate
        if channels is None:
            raise TypeError('channels must be specified')
        info.channels = channels
        info.format = _format_int(format, subtype, endian)
    elif any((arg is not None for arg in (samplerate, channels, original_format, subtype, endian))):
        raise TypeError("Not allowed for existing files (except 'RAW'): samplerate, channels, format, subtype, endian")
    return info