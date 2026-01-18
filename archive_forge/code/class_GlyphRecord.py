from fontTools.misc import sstruct
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
import pdb
import struct
class GlyphRecord(object):

    def __init__(self):
        self.glyphID = -1
        self.nMetaEntry = -1
        self.offset = -1
        self.stringRecs = []

    def toXML(self, writer, ttFont):
        writer.begintag('GlyphRecord')
        writer.newline()
        writer.simpletag('glyphID', value=self.glyphID)
        writer.newline()
        writer.simpletag('nMetaEntry', value=self.nMetaEntry)
        writer.newline()
        for stringRec in self.stringRecs:
            stringRec.toXML(writer, ttFont)
        writer.endtag('GlyphRecord')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'StringRecord':
            stringRec = StringRecord()
            self.stringRecs.append(stringRec)
            for element in content:
                if isinstance(element, str):
                    continue
                stringRec.fromXML(name, attrs, content, ttFont)
            stringRec.stringLen = len(stringRec.string)
        else:
            setattr(self, name, safeEval(attrs['value']))

    def compile(self, parentTable):
        data = sstruct.pack(METAGlyphRecordFormat, self)
        if parentTable.metaFlags == 0:
            datum = struct.pack('>H', self.offset)
        elif parentTable.metaFlags == 1:
            datum = struct.pack('>L', self.offset)
        data = data + datum
        return data

    def __repr__(self):
        return 'GlyphRecord[ glyphID: ' + str(self.glyphID) + ', nMetaEntry: ' + str(self.nMetaEntry) + ', offset: ' + str(self.offset) + ' ]'