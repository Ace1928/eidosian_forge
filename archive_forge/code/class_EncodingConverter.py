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
class EncodingConverter(SimpleConverter):

    def _read(self, parent, value):
        if value == 0:
            return 'StandardEncoding'
        elif value == 1:
            return 'ExpertEncoding'
        else:
            assert value > 1
            file = parent.file
            file.seek(value)
            log.log(DEBUG, 'loading Encoding at %s', value)
            fmt = readCard8(file)
            haveSupplement = fmt & 128
            if haveSupplement:
                raise NotImplementedError('Encoding supplements are not yet supported')
            fmt = fmt & 127
            if fmt == 0:
                encoding = parseEncoding0(parent.charset, file, haveSupplement, parent.strings)
            elif fmt == 1:
                encoding = parseEncoding1(parent.charset, file, haveSupplement, parent.strings)
            return encoding

    def write(self, parent, value):
        if value == 'StandardEncoding':
            return 0
        elif value == 'ExpertEncoding':
            return 1
        return 0

    def xmlWrite(self, xmlWriter, name, value):
        if value in ('StandardEncoding', 'ExpertEncoding'):
            xmlWriter.simpletag(name, name=value)
            xmlWriter.newline()
            return
        xmlWriter.begintag(name)
        xmlWriter.newline()
        for code in range(len(value)):
            glyphName = value[code]
            if glyphName != '.notdef':
                xmlWriter.simpletag('map', code=hex(code), name=glyphName)
                xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def xmlRead(self, name, attrs, content, parent):
        if 'name' in attrs:
            return attrs['name']
        encoding = ['.notdef'] * 256
        for element in content:
            if isinstance(element, str):
                continue
            name, attrs, content = element
            code = safeEval(attrs['code'])
            glyphName = attrs['name']
            encoding[code] = glyphName
        return encoding