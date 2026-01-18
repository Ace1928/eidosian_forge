import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class SfntName(object):
    """
    SfntName wrapper

    A structure used to model an SFNT 'name' table entry.
    """

    def __init__(self, name):
        """
        Create a new SfntName object.

        :param name : SFNT 'name' table entry.

        """
        self._FT_SfntName = name
    platform_id = property(lambda self: self._FT_SfntName.platform_id, doc="The platform ID for 'string'.")
    encoding_id = property(lambda self: self._FT_SfntName.encoding_id, doc="The encoding ID for 'string'.")
    language_id = property(lambda self: self._FT_SfntName.language_id, doc="The language ID for 'string'.")
    name_id = property(lambda self: self._FT_SfntName.name_id, doc="An identifier for 'string'.")
    string_len = property(lambda self: self._FT_SfntName.string_len, doc="The length of 'string' in bytes.")

    def _get_string(self):
        s = string_at(self._FT_SfntName.string, self._FT_SfntName.string_len)
        return s
    string = property(_get_string, doc="The 'name' string. Note that its format differs depending on\n                the (platform,encoding) pair. It can be a Pascal String, a\n                UTF-16 one, etc.\n\n                Generally speaking, the string is not zero-terminated. Please\n                refer to the TrueType specification for details.")