from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
class CFFFontSet(object):
    """A CFF font "file" can contain more than one font, although this is
    extremely rare (and not allowed within OpenType fonts).

    This class is the entry point for parsing a CFF table. To actually
    manipulate the data inside the CFF font, you will want to access the
    ``CFFFontSet``'s :class:`TopDict` object. To do this, a ``CFFFontSet``
    object can either be treated as a dictionary (with appropriate
    ``keys()`` and ``values()`` methods) mapping font names to :class:`TopDict`
    objects, or as a list.

    .. code:: python

            from fontTools import ttLib
            tt = ttLib.TTFont("Tests/cffLib/data/LinLibertine_RBI.otf")
            tt["CFF "].cff
            # <fontTools.cffLib.CFFFontSet object at 0x101e24c90>
            tt["CFF "].cff[0] # Here's your actual font data
            # <fontTools.cffLib.TopDict object at 0x1020f1fd0>

    """

    def decompile(self, file, otFont, isCFF2=None):
        """Parse a binary CFF file into an internal representation. ``file``
        should be a file handle object. ``otFont`` is the top-level
        :py:class:`fontTools.ttLib.ttFont.TTFont` object containing this CFF file.

        If ``isCFF2`` is passed and set to ``True`` or ``False``, then the
        library makes an assertion that the CFF header is of the appropriate
        version.
        """
        self.otFont = otFont
        sstruct.unpack(cffHeaderFormat, file.read(3), self)
        if isCFF2 is not None:
            expected_major = 2 if isCFF2 else 1
            if self.major != expected_major:
                raise ValueError("Invalid CFF 'major' version: expected %d, found %d" % (expected_major, self.major))
        else:
            assert self.major in (1, 2), 'Unknown CFF format'
            isCFF2 = self.major == 2
        if not isCFF2:
            self.offSize = struct.unpack('B', file.read(1))[0]
            file.seek(self.hdrSize)
            self.fontNames = list((tostr(s) for s in Index(file, isCFF2=isCFF2)))
            self.topDictIndex = TopDictIndex(file, isCFF2=isCFF2)
            self.strings = IndexedStrings(file)
        else:
            self.topDictSize = struct.unpack('>H', file.read(2))[0]
            file.seek(self.hdrSize)
            self.fontNames = ['CFF2Font']
            cff2GetGlyphOrder = otFont.getGlyphOrder
            self.topDictIndex = TopDictIndex(file, cff2GetGlyphOrder, self.topDictSize, isCFF2=isCFF2)
            self.strings = None
        self.GlobalSubrs = GlobalSubrsIndex(file, isCFF2=isCFF2)
        self.topDictIndex.strings = self.strings
        self.topDictIndex.GlobalSubrs = self.GlobalSubrs

    def __len__(self):
        return len(self.fontNames)

    def keys(self):
        return list(self.fontNames)

    def values(self):
        return self.topDictIndex

    def __getitem__(self, nameOrIndex):
        """Return TopDict instance identified by name (str) or index (int
        or any object that implements `__index__`).
        """
        if hasattr(nameOrIndex, '__index__'):
            index = nameOrIndex.__index__()
        elif isinstance(nameOrIndex, str):
            name = nameOrIndex
            try:
                index = self.fontNames.index(name)
            except ValueError:
                raise KeyError(nameOrIndex)
        else:
            raise TypeError(nameOrIndex)
        return self.topDictIndex[index]

    def compile(self, file, otFont, isCFF2=None):
        """Write the object back into binary representation onto the given file.
        ``file`` should be a file handle object. ``otFont`` is the top-level
        :py:class:`fontTools.ttLib.ttFont.TTFont` object containing this CFF file.

        If ``isCFF2`` is passed and set to ``True`` or ``False``, then the
        library makes an assertion that the CFF header is of the appropriate
        version.
        """
        self.otFont = otFont
        if isCFF2 is not None:
            expected_major = 2 if isCFF2 else 1
            if self.major != expected_major:
                raise ValueError("Invalid CFF 'major' version: expected %d, found %d" % (expected_major, self.major))
        else:
            assert self.major in (1, 2), 'Unknown CFF format'
            isCFF2 = self.major == 2
        if otFont.recalcBBoxes and (not isCFF2):
            for topDict in self.topDictIndex:
                topDict.recalcFontBBox()
        if not isCFF2:
            strings = IndexedStrings()
        else:
            strings = None
        writer = CFFWriter(isCFF2)
        topCompiler = self.topDictIndex.getCompiler(strings, self, isCFF2=isCFF2)
        if isCFF2:
            self.hdrSize = 5
            writer.add(sstruct.pack(cffHeaderFormat, self))
            self.topDictSize = topCompiler.getDataLength()
            writer.add(struct.pack('>H', self.topDictSize))
        else:
            self.hdrSize = 4
            self.offSize = 4
            writer.add(sstruct.pack(cffHeaderFormat, self))
            writer.add(struct.pack('B', self.offSize))
        if not isCFF2:
            fontNames = Index()
            for name in self.fontNames:
                fontNames.append(name)
            writer.add(fontNames.getCompiler(strings, self, isCFF2=isCFF2))
        writer.add(topCompiler)
        if not isCFF2:
            writer.add(strings.getCompiler())
        writer.add(self.GlobalSubrs.getCompiler(strings, self, isCFF2=isCFF2))
        for topDict in self.topDictIndex:
            if not hasattr(topDict, 'charset') or topDict.charset is None:
                charset = otFont.getGlyphOrder()
                topDict.charset = charset
        children = topCompiler.getChildren(strings)
        for child in children:
            writer.add(child)
        writer.toFile(file)

    def toXML(self, xmlWriter):
        """Write the object into XML representation onto the given
        :class:`fontTools.misc.xmlWriter.XMLWriter`.

        .. code:: python

                writer = xmlWriter.XMLWriter(sys.stdout)
                tt["CFF "].cff.toXML(writer)

        """
        xmlWriter.simpletag('major', value=self.major)
        xmlWriter.newline()
        xmlWriter.simpletag('minor', value=self.minor)
        xmlWriter.newline()
        for fontName in self.fontNames:
            xmlWriter.begintag('CFFFont', name=tostr(fontName))
            xmlWriter.newline()
            font = self[fontName]
            font.toXML(xmlWriter)
            xmlWriter.endtag('CFFFont')
            xmlWriter.newline()
        xmlWriter.newline()
        xmlWriter.begintag('GlobalSubrs')
        xmlWriter.newline()
        self.GlobalSubrs.toXML(xmlWriter)
        xmlWriter.endtag('GlobalSubrs')
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, otFont=None):
        """Reads data from the XML element into the ``CFFFontSet`` object."""
        self.otFont = otFont
        if not hasattr(self, 'major'):
            self.major = 1
        if not hasattr(self, 'minor'):
            self.minor = 0
        if name == 'CFFFont':
            if self.major == 1:
                if not hasattr(self, 'offSize'):
                    self.offSize = 4
                if not hasattr(self, 'hdrSize'):
                    self.hdrSize = 4
                if not hasattr(self, 'GlobalSubrs'):
                    self.GlobalSubrs = GlobalSubrsIndex()
                if not hasattr(self, 'fontNames'):
                    self.fontNames = []
                    self.topDictIndex = TopDictIndex()
                fontName = attrs['name']
                self.fontNames.append(fontName)
                topDict = TopDict(GlobalSubrs=self.GlobalSubrs)
                topDict.charset = None
            elif self.major == 2:
                if not hasattr(self, 'hdrSize'):
                    self.hdrSize = 5
                if not hasattr(self, 'GlobalSubrs'):
                    self.GlobalSubrs = GlobalSubrsIndex()
                if not hasattr(self, 'fontNames'):
                    self.fontNames = ['CFF2Font']
                cff2GetGlyphOrder = self.otFont.getGlyphOrder
                topDict = TopDict(GlobalSubrs=self.GlobalSubrs, cff2GetGlyphOrder=cff2GetGlyphOrder)
                self.topDictIndex = TopDictIndex(None, cff2GetGlyphOrder)
            self.topDictIndex.append(topDict)
            for element in content:
                if isinstance(element, str):
                    continue
                name, attrs, content = element
                topDict.fromXML(name, attrs, content)
            if hasattr(topDict, 'VarStore') and topDict.FDArray[0].vstore is None:
                fdArray = topDict.FDArray
                for fontDict in fdArray:
                    if hasattr(fontDict, 'Private'):
                        fontDict.Private.vstore = topDict.VarStore
        elif name == 'GlobalSubrs':
            subrCharStringClass = psCharStrings.T2CharString
            if not hasattr(self, 'GlobalSubrs'):
                self.GlobalSubrs = GlobalSubrsIndex()
            for element in content:
                if isinstance(element, str):
                    continue
                name, attrs, content = element
                subr = subrCharStringClass()
                subr.fromXML(name, attrs, content)
                self.GlobalSubrs.append(subr)
        elif name == 'major':
            self.major = int(attrs['value'])
        elif name == 'minor':
            self.minor = int(attrs['value'])

    def convertCFFToCFF2(self, otFont):
        """Converts this object from CFF format to CFF2 format. This conversion
        is done 'in-place'. The conversion cannot be reversed.

        This assumes a decompiled CFF table. (i.e. that the object has been
        filled via :meth:`decompile`.)"""
        self.major = 2
        cff2GetGlyphOrder = self.otFont.getGlyphOrder
        topDictData = TopDictIndex(None, cff2GetGlyphOrder)
        topDictData.items = self.topDictIndex.items
        self.topDictIndex = topDictData
        topDict = topDictData[0]
        if hasattr(topDict, 'Private'):
            privateDict = topDict.Private
        else:
            privateDict = None
        opOrder = buildOrder(topDictOperators2)
        topDict.order = opOrder
        topDict.cff2GetGlyphOrder = cff2GetGlyphOrder
        for entry in topDictOperators:
            key = entry[1]
            if key not in opOrder:
                if key in topDict.rawDict:
                    del topDict.rawDict[key]
                if hasattr(topDict, key):
                    delattr(topDict, key)
        if not hasattr(topDict, 'FDArray'):
            fdArray = topDict.FDArray = FDArrayIndex()
            fdArray.strings = None
            fdArray.GlobalSubrs = topDict.GlobalSubrs
            topDict.GlobalSubrs.fdArray = fdArray
            charStrings = topDict.CharStrings
            if charStrings.charStringsAreIndexed:
                charStrings.charStringsIndex.fdArray = fdArray
            else:
                charStrings.fdArray = fdArray
            fontDict = FontDict()
            fontDict.setCFF2(True)
            fdArray.append(fontDict)
            fontDict.Private = privateDict
            privateOpOrder = buildOrder(privateDictOperators2)
            for entry in privateDictOperators:
                key = entry[1]
                if key not in privateOpOrder:
                    if key in privateDict.rawDict:
                        del privateDict.rawDict[key]
                    if hasattr(privateDict, key):
                        delattr(privateDict, key)
        else:
            fdArray = topDict.FDArray
            privateOpOrder = buildOrder(privateDictOperators2)
            for fontDict in fdArray:
                fontDict.setCFF2(True)
                for key in fontDict.rawDict.keys():
                    if key not in fontDict.order:
                        del fontDict.rawDict[key]
                        if hasattr(fontDict, key):
                            delattr(fontDict, key)
                privateDict = fontDict.Private
                for entry in privateDictOperators:
                    key = entry[1]
                    if key not in privateOpOrder:
                        if key in privateDict.rawDict:
                            del privateDict.rawDict[key]
                        if hasattr(privateDict, key):
                            delattr(privateDict, key)
        file = BytesIO()
        self.compile(file, otFont, isCFF2=True)
        file.seek(0)
        self.decompile(file, otFont, isCFF2=True)

    def desubroutinize(self):
        for fontName in self.fontNames:
            font = self[fontName]
            cs = font.CharStrings
            for g in font.charset:
                c, _ = cs.getItemAndSelector(g)
                c.decompile()
                subrs = getattr(c.private, 'Subrs', [])
                decompiler = _DesubroutinizingT2Decompiler(subrs, c.globalSubrs, c.private)
                decompiler.execute(c)
                c.program = c._desubroutinized
                del c._desubroutinized
            if hasattr(font, 'FDArray'):
                for fd in font.FDArray:
                    pd = fd.Private
                    if hasattr(pd, 'Subrs'):
                        del pd.Subrs
                    if 'Subrs' in pd.rawDict:
                        del pd.rawDict['Subrs']
            else:
                pd = font.Private
                if hasattr(pd, 'Subrs'):
                    del pd.Subrs
                if 'Subrs' in pd.rawDict:
                    del pd.rawDict['Subrs']
        self.GlobalSubrs.clear()