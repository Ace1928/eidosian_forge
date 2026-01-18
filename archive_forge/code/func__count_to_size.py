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
def _count_to_size(count):
    if count < 1 << 8:
        return 1
    elif count < 1 << 16:
        return 2
    elif count < 1 << 32:
        return 4
    else:
        return 8