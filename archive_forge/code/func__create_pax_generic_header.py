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
@classmethod
def _create_pax_generic_header(cls, pax_headers, type, encoding):
    """Return a POSIX.1-2008 extended or global header sequence
           that contains a list of keyword, value pairs. The values
           must be strings.
        """
    binary = False
    for keyword, value in pax_headers.items():
        try:
            value.encode('utf-8', 'strict')
        except UnicodeEncodeError:
            binary = True
            break
    records = b''
    if binary:
        records += b'21 hdrcharset=BINARY\n'
    for keyword, value in pax_headers.items():
        keyword = keyword.encode('utf-8')
        if binary:
            value = value.encode(encoding, 'surrogateescape')
        else:
            value = value.encode('utf-8')
        l = len(keyword) + len(value) + 3
        n = p = 0
        while True:
            n = l + len(str(p))
            if n == p:
                break
            p = n
        records += bytes(str(p), 'ascii') + b' ' + keyword + b'=' + value + b'\n'
    info = {}
    info['name'] = '././@PaxHeader'
    info['type'] = type
    info['size'] = len(records)
    info['magic'] = POSIX_MAGIC
    return cls._create_header(info, USTAR_FORMAT, 'ascii', 'replace') + cls._create_payload(records)