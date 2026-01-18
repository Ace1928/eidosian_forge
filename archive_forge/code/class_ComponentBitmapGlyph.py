from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class ComponentBitmapGlyph(BitmapGlyph):

    def toXML(self, strikeIndex, glyphName, writer, ttFont):
        writer.begintag(self.__class__.__name__, [('name', glyphName)])
        writer.newline()
        self.writeMetrics(writer, ttFont)
        writer.begintag('components')
        writer.newline()
        for curComponent in self.componentArray:
            curComponent.toXML(writer, ttFont)
        writer.endtag('components')
        writer.newline()
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.readMetrics(name, attrs, content, ttFont)
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attr, content = element
            if name == 'components':
                self.componentArray = []
                for compElement in content:
                    if not isinstance(compElement, tuple):
                        continue
                    name, attrs, content = compElement
                    if name == 'ebdtComponent':
                        curComponent = EbdtComponent()
                        curComponent.fromXML(name, attrs, content, ttFont)
                        self.componentArray.append(curComponent)
                    else:
                        log.warning("'%s' being ignored in component array.", name)