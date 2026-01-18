from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class EbdtComponent(object):

    def toXML(self, writer, ttFont):
        writer.begintag('ebdtComponent', [('name', self.name)])
        writer.newline()
        for componentName in sstruct.getformat(ebdtComponentFormat)[1][1:]:
            writer.simpletag(componentName, value=getattr(self, componentName))
            writer.newline()
        writer.endtag('ebdtComponent')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.name = attrs['name']
        componentNames = set(sstruct.getformat(ebdtComponentFormat)[1][1:])
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name in componentNames:
                vars(self)[name] = safeEval(attrs['value'])
            else:
                log.warning("unknown name '%s' being ignored by EbdtComponent.", name)