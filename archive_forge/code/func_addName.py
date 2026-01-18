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
def addName(self, string, platforms=((1, 0, 0), (3, 1, 1033)), minNameID=255):
    """Add a new name record containing 'string' for each (platformID, platEncID,
        langID) tuple specified in the 'platforms' list.

        The nameID is assigned in the range between 'minNameID'+1 and 32767 (inclusive),
        following the last nameID in the name table.
        If no 'platforms' are specified, two English name records are added, one for the
        Macintosh (platformID=0), and one for the Windows platform (3).

        The 'string' must be a Unicode string, so it can be encoded with different,
        platform-specific encodings.

        Return the new nameID.
        """
    assert len(platforms) > 0, "'platforms' must contain at least one (platformID, platEncID, langID) tuple"
    if not hasattr(self, 'names'):
        self.names = []
    if not isinstance(string, str):
        raise TypeError('expected str, found %s: %r' % (type(string).__name__, string))
    nameID = self._findUnusedNameID(minNameID + 1)
    for platformID, platEncID, langID in platforms:
        self.names.append(makeName(string, nameID, platformID, platEncID, langID))
    return nameID