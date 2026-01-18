from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
@staticmethod
def _create_header(info, format, encoding, errors):
    """Return a header block. info is a dictionary with file
           information, format must be one of the *_FORMAT constants.
        """
    has_device_fields = info.get('type') in (CHRTYPE, BLKTYPE)
    if has_device_fields:
        devmajor = itn(info.get('devmajor', 0), 8, format)
        devminor = itn(info.get('devminor', 0), 8, format)
    else:
        devmajor = stn('', 8, encoding, errors)
        devminor = stn('', 8, encoding, errors)
    filetype = info.get('type', REGTYPE)
    if filetype is None:
        raise ValueError('TarInfo.type must not be None')
    parts = [stn(info.get('name', ''), 100, encoding, errors), itn(info.get('mode', 0) & 4095, 8, format), itn(info.get('uid', 0), 8, format), itn(info.get('gid', 0), 8, format), itn(info.get('size', 0), 12, format), itn(info.get('mtime', 0), 12, format), b'        ', filetype, stn(info.get('linkname', ''), 100, encoding, errors), info.get('magic', POSIX_MAGIC), stn(info.get('uname', ''), 32, encoding, errors), stn(info.get('gname', ''), 32, encoding, errors), devmajor, devminor, stn(info.get('prefix', ''), 155, encoding, errors)]
    buf = struct.pack('%ds' % BLOCKSIZE, b''.join(parts))
    chksum = calc_chksums(buf[-BLOCKSIZE:])[0]
    buf = buf[:-364] + bytes('%06o\x00' % chksum, 'ascii') + buf[-357:]
    return buf