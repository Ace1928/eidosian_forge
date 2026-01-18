from fontTools.config import Config
from fontTools.misc import xmlWriter
from fontTools.misc.configTools import AbstractConfig
from fontTools.misc.textTools import Tag, byteord, tostr
from fontTools.misc.loggingTools import deprecateArgument
from fontTools.ttLib import TTLibError
from fontTools.ttLib.ttGlyphSet import _TTGlyph, _TTGlyphSetCFF, _TTGlyphSetGlyf
from fontTools.ttLib.sfnt import SFNTReader, SFNTWriter
from io import BytesIO, StringIO, UnsupportedOperation
import os
import logging
import traceback
def _saveXML(self, writer, writeVersion=True, quiet=None, tables=None, skipTables=None, splitTables=False, splitGlyphs=False, disassembleInstructions=True, bitmapGlyphDataFormat='raw'):
    if quiet is not None:
        deprecateArgument('quiet', 'configure logging instead')
    self.disassembleInstructions = disassembleInstructions
    self.bitmapGlyphDataFormat = bitmapGlyphDataFormat
    if not tables:
        tables = list(self.keys())
        if 'GlyphOrder' not in tables:
            tables = ['GlyphOrder'] + tables
        if skipTables:
            for tag in skipTables:
                if tag in tables:
                    tables.remove(tag)
    numTables = len(tables)
    if writeVersion:
        from fontTools import version
        version = '.'.join(version.split('.')[:2])
        writer.begintag('ttFont', sfntVersion=repr(tostr(self.sfntVersion))[1:-1], ttLibVersion=version)
    else:
        writer.begintag('ttFont', sfntVersion=repr(tostr(self.sfntVersion))[1:-1])
    writer.newline()
    splitTables = splitTables or splitGlyphs
    if not splitTables:
        writer.newline()
    else:
        path, ext = os.path.splitext(writer.filename)
    for i in range(numTables):
        tag = tables[i]
        if splitTables:
            tablePath = path + '.' + tagToIdentifier(tag) + ext
            tableWriter = xmlWriter.XMLWriter(tablePath, newlinestr=writer.newlinestr)
            tableWriter.begintag('ttFont', ttLibVersion=version)
            tableWriter.newline()
            tableWriter.newline()
            writer.simpletag(tagToXML(tag), src=os.path.basename(tablePath))
            writer.newline()
        else:
            tableWriter = writer
        self._tableToXML(tableWriter, tag, splitGlyphs=splitGlyphs)
        if splitTables:
            tableWriter.endtag('ttFont')
            tableWriter.newline()
            tableWriter.close()
    writer.endtag('ttFont')
    writer.newline()