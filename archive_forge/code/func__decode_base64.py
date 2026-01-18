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
def _decode_base64(s):
    if isinstance(s, str):
        return binascii.a2b_base64(s.encode('utf-8'))
    else:
        return binascii.a2b_base64(s)