from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
class TFMException(Exception):

    def __init__(self, message):
        super().__init__(message)