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
class table__n_a_m_e(DefaultTable.DefaultTable):
    dependencies = ['ltag']

    def decompile(self, data, ttFont):
        format, n, stringOffset = struct.unpack(b'>HHH', data[:6])
        expectedStringOffset = 6 + n * nameRecordSize
        if stringOffset != expectedStringOffset:
            log.error("'name' table stringOffset incorrect. Expected: %s; Actual: %s", expectedStringOffset, stringOffset)
        stringData = data[stringOffset:]
        data = data[6:]
        self.names = []
        for i in range(n):
            if len(data) < 12:
                log.error('skipping malformed name record #%d', i)
                continue
            name, data = sstruct.unpack2(nameRecordFormat, data, NameRecord())
            name.string = stringData[name.offset:name.offset + name.length]
            if name.offset + name.length > len(stringData):
                log.error('skipping malformed name record #%d', i)
                continue
            assert len(name.string) == name.length
            del name.offset, name.length
            self.names.append(name)

    def compile(self, ttFont):
        if not hasattr(self, 'names'):
            self.names = []
        names = self.names
        names.sort()
        stringData = b''
        format = 0
        n = len(names)
        stringOffset = 6 + n * sstruct.calcsize(nameRecordFormat)
        data = struct.pack(b'>HHH', format, n, stringOffset)
        lastoffset = 0
        done = {}
        for name in names:
            string = name.toBytes()
            if string in done:
                name.offset, name.length = done[string]
            else:
                name.offset, name.length = done[string] = (len(stringData), len(string))
                stringData = bytesjoin([stringData, string])
            data = data + sstruct.pack(nameRecordFormat, name)
        return data + stringData

    def toXML(self, writer, ttFont):
        for name in self.names:
            name.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name != 'namerecord':
            return
        if not hasattr(self, 'names'):
            self.names = []
        name = NameRecord()
        self.names.append(name)
        name.fromXML(name, attrs, content, ttFont)

    def getName(self, nameID, platformID, platEncID, langID=None):
        for namerecord in self.names:
            if namerecord.nameID == nameID and namerecord.platformID == platformID and (namerecord.platEncID == platEncID):
                if langID is None or namerecord.langID == langID:
                    return namerecord
        return None

    def getDebugName(self, nameID):
        englishName = someName = None
        for name in self.names:
            if name.nameID != nameID:
                continue
            try:
                unistr = name.toUnicode()
            except UnicodeDecodeError:
                continue
            someName = unistr
            if (name.platformID, name.langID) in ((1, 0), (3, 1033)):
                englishName = unistr
                break
        if englishName:
            return englishName
        elif someName:
            return someName
        else:
            return None

    def getFirstDebugName(self, nameIDs):
        for nameID in nameIDs:
            name = self.getDebugName(nameID)
            if name is not None:
                return name
        return None

    def getBestFamilyName(self):
        return self.getFirstDebugName((21, 16, 1))

    def getBestSubFamilyName(self):
        return self.getFirstDebugName((22, 17, 2))

    def getBestFullName(self):
        for nameIDs in ((21, 22), (16, 17), (1, 2), (4,), (6,)):
            if len(nameIDs) == 2:
                name_fam = self.getDebugName(nameIDs[0])
                name_subfam = self.getDebugName(nameIDs[1])
                if None in [name_fam, name_subfam]:
                    continue
                name = f'{name_fam} {name_subfam}'
                if name_subfam.lower() == 'regular':
                    name = f'{name_fam}'
                return name
            else:
                name = self.getDebugName(nameIDs[0])
                if name is not None:
                    return name
        return None

    def setName(self, string, nameID, platformID, platEncID, langID):
        """Set the 'string' for the name record identified by 'nameID', 'platformID',
        'platEncID' and 'langID'. If a record with that nameID doesn't exist, create it
        and append to the name table.

        'string' can be of type `str` (`unicode` in PY2) or `bytes`. In the latter case,
        it is assumed to be already encoded with the correct plaform-specific encoding
        identified by the (platformID, platEncID, langID) triplet. A warning is issued
        to prevent unexpected results.
        """
        if not hasattr(self, 'names'):
            self.names = []
        if not isinstance(string, str):
            if isinstance(string, bytes):
                log.warning("name string is bytes, ensure it's correctly encoded: %r", string)
            else:
                raise TypeError('expected unicode or bytes, found %s: %r' % (type(string).__name__, string))
        namerecord = self.getName(nameID, platformID, platEncID, langID)
        if namerecord:
            namerecord.string = string
        else:
            self.names.append(makeName(string, nameID, platformID, platEncID, langID))

    def removeNames(self, nameID=None, platformID=None, platEncID=None, langID=None):
        """Remove any name records identified by the given combination of 'nameID',
        'platformID', 'platEncID' and 'langID'.
        """
        args = {argName: argValue for argName, argValue in (('nameID', nameID), ('platformID', platformID), ('platEncID', platEncID), ('langID', langID)) if argValue is not None}
        if not args:
            return
        self.names = [rec for rec in self.names if any((argValue != getattr(rec, argName) for argName, argValue in args.items()))]

    @staticmethod
    def removeUnusedNames(ttFont):
        """Remove any name records which are not in NameID range 0-255 and not utilized
        within the font itself."""
        visitor = NameRecordVisitor()
        visitor.visit(ttFont)
        toDelete = set()
        for record in ttFont['name'].names:
            if record.nameID < 256:
                continue
            if record.nameID not in visitor.seen:
                toDelete.add(record.nameID)
        for nameID in toDelete:
            ttFont['name'].removeNames(nameID)
        return toDelete

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

    def findMultilingualName(self, names, windows=True, mac=True, minNameID=0, ttFont=None):
        """Return the name ID of an existing multilingual name that
        matches the 'names' dictionary, or None if not found.

        'names' is a dictionary with the name in multiple languages,
        such as {'en': 'Pale', 'de': 'Blaß', 'de-CH': 'Blass'}.
        The keys can be arbitrary IETF BCP 47 language codes;
        the values are Unicode strings.

        If 'windows' is True, the returned name ID is guaranteed
        exist for all requested languages for platformID=3 and
        platEncID=1.
        If 'mac' is True, the returned name ID is guaranteed to exist
        for all requested languages for platformID=1 and platEncID=0.

        The returned name ID will not be less than the 'minNameID'
        argument.
        """
        reqNameSet = set()
        for lang, name in sorted(names.items()):
            if windows:
                windowsName = _makeWindowsName(name, None, lang)
                if windowsName is not None:
                    reqNameSet.add((windowsName.string, windowsName.platformID, windowsName.platEncID, windowsName.langID))
            if mac:
                macName = _makeMacName(name, None, lang, ttFont)
                if macName is not None:
                    reqNameSet.add((macName.string, macName.platformID, macName.platEncID, macName.langID))
        matchingNames = dict()
        for name in self.names:
            try:
                key = (name.toUnicode(), name.platformID, name.platEncID, name.langID)
            except UnicodeDecodeError:
                continue
            if key in reqNameSet and name.nameID >= minNameID:
                nameSet = matchingNames.setdefault(name.nameID, set())
                nameSet.add(key)
        for nameID, nameSet in sorted(matchingNames.items()):
            if nameSet == reqNameSet:
                return nameID
        return None

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