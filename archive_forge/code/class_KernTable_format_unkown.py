from fontTools.ttLib import getSearchRange
from fontTools.misc.textTools import safeEval, readHex
from fontTools.misc.fixedTools import fixedToFloat as fi2fl, floatToFixed as fl2fi
from . import DefaultTable
import struct
import sys
import array
import logging
class KernTable_format_unkown(object):

    def __init__(self, format):
        self.format = format

    def decompile(self, data, ttFont):
        self.data = data

    def compile(self, ttFont):
        return self.data

    def toXML(self, writer, ttFont):
        writer.begintag('kernsubtable', format=self.format)
        writer.newline()
        writer.comment("unknown 'kern' subtable format")
        writer.newline()
        writer.dumphex(self.data)
        writer.endtag('kernsubtable')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.decompile(readHex(content), ttFont)