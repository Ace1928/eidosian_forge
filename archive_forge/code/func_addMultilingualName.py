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
def addMultilingualName(self, names, ttFont=None, nameID=None, windows=True, mac=True, minNameID=0):
    """Add a multilingual name, returning its name ID

        'names' is a dictionary with the name in multiple languages,
        such as {'en': 'Pale', 'de': 'Blaß', 'de-CH': 'Blass'}.
        The keys can be arbitrary IETF BCP 47 language codes;
        the values are Unicode strings.

        'ttFont' is the TTFont to which the names are added, or None.
        If present, the font's 'ltag' table can get populated
        to store exotic language codes, which allows encoding
        names that otherwise cannot get encoded at all.

        'nameID' is the name ID to be used, or None to let the library
        find an existing set of name records that match, or pick an
        unused name ID.

        If 'windows' is True, a platformID=3 name record will be added.
        If 'mac' is True, a platformID=1 name record will be added.

        If the 'nameID' argument is None, the created nameID will not
        be less than the 'minNameID' argument.
        """
    if not hasattr(self, 'names'):
        self.names = []
    if nameID is None:
        nameID = self.findMultilingualName(names, windows=windows, mac=mac, minNameID=minNameID, ttFont=ttFont)
        if nameID is not None:
            return nameID
        nameID = self._findUnusedNameID()
    for lang, name in sorted(names.items()):
        if windows:
            windowsName = _makeWindowsName(name, nameID, lang)
            if windowsName is not None:
                self.names.append(windowsName)
            else:
                mac = True
        if mac:
            macName = _makeMacName(name, nameID, lang, ttFont)
            if macName is not None:
                self.names.append(macName)
    return nameID