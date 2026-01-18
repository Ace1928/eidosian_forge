import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
class _Example:

    def __init__(self, value):
        self.value = value