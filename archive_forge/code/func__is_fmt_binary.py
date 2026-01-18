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
def _is_fmt_binary(header):
    return header[:8] == b'bplist00'