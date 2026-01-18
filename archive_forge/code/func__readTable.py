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
def _readTable(self, tag):
    log.debug("Reading '%s' table from disk", tag)
    data = self.reader[tag]
    if self._tableCache is not None:
        table = self._tableCache.get((tag, data))
        if table is not None:
            return table
    tableClass = getTableClass(tag)
    table = tableClass(tag)
    self.tables[tag] = table
    log.debug("Decompiling '%s' table", tag)
    try:
        table.decompile(data, self)
    except Exception:
        if not self.ignoreDecompileErrors:
            raise
        log.exception("An exception occurred during the decompilation of the '%s' table", tag)
        from .tables.DefaultTable import DefaultTable
        file = StringIO()
        traceback.print_exc(file=file)
        table = DefaultTable(tag)
        table.ERROR = file.getvalue()
        self.tables[tag] = table
        table.decompile(data, self)
    if self._tableCache is not None:
        self._tableCache[tag, data] = table
    return table