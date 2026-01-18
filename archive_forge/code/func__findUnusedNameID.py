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
def _findUnusedNameID(self, minNameID=256):
    """Finds an unused name id.

        The nameID is assigned in the range between 'minNameID' and 32767 (inclusive),
        following the last nameID in the name table.
        """
    names = getattr(self, 'names', [])
    nameID = 1 + max([n.nameID for n in names] + [minNameID - 1])
    if nameID > 32767:
        raise ValueError('nameID must be less than 32768')
    return nameID