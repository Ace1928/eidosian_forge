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
def _writeTable(self, tag, writer, done, tableCache=None):
    """Internal helper function for self.save(). Keeps track of
        inter-table dependencies.
        """
    if tag in done:
        return
    tableClass = getTableClass(tag)
    for masterTable in tableClass.dependencies:
        if masterTable not in done:
            if masterTable in self:
                self._writeTable(masterTable, writer, done, tableCache)
            else:
                done.append(masterTable)
    done.append(tag)
    tabledata = self.getTableData(tag)
    if tableCache is not None:
        entry = tableCache.get((Tag(tag), tabledata))
        if entry is not None:
            log.debug("reusing '%s' table", tag)
            writer.setEntry(tag, entry)
            return
    log.debug("Writing '%s' table to disk", tag)
    writer[tag] = tabledata
    if tableCache is not None:
        tableCache[Tag(tag), tabledata] = writer[tag]