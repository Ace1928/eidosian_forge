import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _format_info(format_int, format_flag=_snd.SFC_GET_FORMAT_INFO):
    """Return the ID and short description of a given format."""
    format_info = _ffi.new('SF_FORMAT_INFO*')
    format_info.format = format_int
    _snd.sf_command(_ffi.NULL, format_flag, format_info, _ffi.sizeof('SF_FORMAT_INFO'))
    name = format_info.name
    return (_format_str(format_info.format), _ffi.string(name).decode('utf-8', 'replace') if name else '')