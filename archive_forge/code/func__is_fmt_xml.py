import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate
def _is_fmt_xml(header):
    prefixes = (b'<?xml', b'<plist')
    for pfx in prefixes:
        if header.startswith(pfx):
            return True
    for bom, encoding in ((codecs.BOM_UTF8, 'utf-8'), (codecs.BOM_UTF16_BE, 'utf-16-be'), (codecs.BOM_UTF16_LE, 'utf-16-le')):
        if not header.startswith(bom):
            continue
        for start in prefixes:
            prefix = bom + start.decode('ascii').encode(encoding)
            if header[:len(prefix)] == prefix:
                return True
    return False