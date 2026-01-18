from fontTools.misc import sstruct
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
import pdb
import struct
class table_M_E_T_A_(DefaultTable.DefaultTable):
    dependencies = []

    def decompile(self, data, ttFont):
        dummy, newData = sstruct.unpack2(METAHeaderFormat, data, self)
        self.glyphRecords = []
        for i in range(self.nMetaRecs):
            glyphRecord, newData = sstruct.unpack2(METAGlyphRecordFormat, newData, GlyphRecord())
            if self.metaFlags == 0:
                [glyphRecord.offset] = struct.unpack('>H', newData[:2])
                newData = newData[2:]
            elif self.metaFlags == 1:
                [glyphRecord.offset] = struct.unpack('>H', newData[:4])
                newData = newData[4:]
            else:
                assert 0, 'The metaFlags field in the META table header has a value other than 0 or 1 :' + str(self.metaFlags)
            glyphRecord.stringRecs = []
            newData = data[glyphRecord.offset:]
            for j in range(glyphRecord.nMetaEntry):
                stringRec, newData = sstruct.unpack2(METAStringRecordFormat, newData, StringRecord())
                if self.metaFlags == 0:
                    [stringRec.offset] = struct.unpack('>H', newData[:2])
                    newData = newData[2:]
                else:
                    [stringRec.offset] = struct.unpack('>H', newData[:4])
                    newData = newData[4:]
                stringRec.string = data[stringRec.offset:stringRec.offset + stringRec.stringLen]
                glyphRecord.stringRecs.append(stringRec)
            self.glyphRecords.append(glyphRecord)

    def compile(self, ttFont):
        offsetOK = 0
        self.nMetaRecs = len(self.glyphRecords)
        count = 0
        while offsetOK != 1:
            count = count + 1
            if count > 4:
                pdb.set_trace()
            metaData = sstruct.pack(METAHeaderFormat, self)
            stringRecsOffset = len(metaData) + self.nMetaRecs * (6 + 2 * (self.metaFlags & 1))
            stringRecSize = 6 + 2 * (self.metaFlags & 1)
            for glyphRec in self.glyphRecords:
                glyphRec.offset = stringRecsOffset
                if glyphRec.offset > 65535 and self.metaFlags & 1 == 0:
                    self.metaFlags = self.metaFlags + 1
                    offsetOK = -1
                    break
                metaData = metaData + glyphRec.compile(self)
                stringRecsOffset = stringRecsOffset + glyphRec.nMetaEntry * stringRecSize
            if offsetOK == -1:
                offsetOK = 0
                continue
            stringOffset = stringRecsOffset
            for glyphRec in self.glyphRecords:
                assert glyphRec.offset == len(metaData), 'Glyph record offset did not compile correctly! for rec:' + str(glyphRec)
                for stringRec in glyphRec.stringRecs:
                    stringRec.offset = stringOffset
                    if stringRec.offset > 65535 and self.metaFlags & 1 == 0:
                        self.metaFlags = self.metaFlags + 1
                        offsetOK = -1
                        break
                    metaData = metaData + stringRec.compile(self)
                    stringOffset = stringOffset + stringRec.stringLen
            if offsetOK == -1:
                offsetOK = 0
                continue
            if self.metaFlags & 1 == 1 and stringOffset < 65536:
                self.metaFlags = self.metaFlags - 1
                continue
            else:
                offsetOK = 1
            for glyphRec in self.glyphRecords:
                for stringRec in glyphRec.stringRecs:
                    assert stringRec.offset == len(metaData), 'String offset did not compile correctly! for string:' + str(stringRec.string)
                    metaData = metaData + stringRec.string
        return metaData

    def toXML(self, writer, ttFont):
        writer.comment('Lengths and number of entries in this table will be recalculated by the compiler')
        writer.newline()
        formatstring, names, fixes = sstruct.getformat(METAHeaderFormat)
        for name in names:
            value = getattr(self, name)
            writer.simpletag(name, value=value)
            writer.newline()
        for glyphRec in self.glyphRecords:
            glyphRec.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'GlyphRecord':
            if not hasattr(self, 'glyphRecords'):
                self.glyphRecords = []
            glyphRec = GlyphRecord()
            self.glyphRecords.append(glyphRec)
            for element in content:
                if isinstance(element, str):
                    continue
                name, attrs, content = element
                glyphRec.fromXML(name, attrs, content, ttFont)
            glyphRec.offset = -1
            glyphRec.nMetaEntry = len(glyphRec.stringRecs)
        else:
            setattr(self, name, safeEval(attrs['value']))