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
class InvalidFileException(ValueError):

    def __init__(self, message='Invalid file'):
        ValueError.__init__(self, message)