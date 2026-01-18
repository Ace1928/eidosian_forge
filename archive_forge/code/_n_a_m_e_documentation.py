from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import newTable
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools import ttLib
import fontTools.ttLib.tables.otTables as otTables
from fontTools.ttLib.tables import C_P_A_L_
from . import DefaultTable
import struct
import logging
If self.string is a bytes object, return it; otherwise try encoding
        the Unicode string in self.string to bytes using the encoding of this
        entry as returned by self.getEncoding(); Note that self.getEncoding()
        returns 'ascii' if the encoding is unknown to the library.

        If the Unicode string cannot be encoded to bytes in the chosen encoding,
        the error is handled according to the errors parameter to this function,
        which is passed to the underlying encode() function; by default it throws a
        UnicodeEncodeError exception.
        