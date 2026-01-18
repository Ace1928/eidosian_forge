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
def _getGlyphNamesFromCmap(self):
    if self.isLoaded('cmap'):
        cmapLoading = self.tables['cmap']
        del self.tables['cmap']
    else:
        cmapLoading = None
    numGlyphs = int(self['maxp'].numGlyphs)
    glyphOrder = [None] * numGlyphs
    glyphOrder[0] = '.notdef'
    for i in range(1, numGlyphs):
        glyphOrder[i] = 'glyph%.5d' % i
    self.glyphOrder = glyphOrder
    if 'cmap' in self:
        reversecmap = self['cmap'].buildReversed()
    else:
        reversecmap = {}
    useCount = {}
    for i in range(numGlyphs):
        tempName = glyphOrder[i]
        if tempName in reversecmap:
            glyphName = self._makeGlyphName(min(reversecmap[tempName]))
            numUses = useCount[glyphName] = useCount.get(glyphName, 0) + 1
            if numUses > 1:
                glyphName = '%s.alt%d' % (glyphName, numUses - 1)
            glyphOrder[i] = glyphName
    if 'cmap' in self:
        del self.tables['cmap']
        self.glyphOrder = glyphOrder
        if cmapLoading:
            self.tables['cmap'] = cmapLoading