from . import DefaultTable
import struct
class table_T_S_I__0(DefaultTable.DefaultTable):
    dependencies = ['TSI1']

    def decompile(self, data, ttFont):
        numGlyphs = ttFont['maxp'].numGlyphs
        indices = []
        size = struct.calcsize(tsi0Format)
        for i in range(numGlyphs + 5):
            glyphID, textLength, textOffset = fixlongs(*struct.unpack(tsi0Format, data[:size]))
            indices.append((glyphID, textLength, textOffset))
            data = data[size:]
        assert len(data) == 0
        assert indices[-5] == (65534, 0, 2885426996), 'bad magic number'
        self.indices = indices[:-5]
        self.extra_indices = indices[-4:]

    def compile(self, ttFont):
        if not hasattr(self, 'indices'):
            return b''
        data = b''
        for index, textLength, textOffset in self.indices:
            data = data + struct.pack(tsi0Format, index, textLength, textOffset)
        data = data + struct.pack(tsi0Format, 65534, 0, 2885426996)
        for index, textLength, textOffset in self.extra_indices:
            data = data + struct.pack(tsi0Format, index, textLength, textOffset)
        return data

    def set(self, indices, extra_indices):
        self.indices = indices
        self.extra_indices = extra_indices

    def toXML(self, writer, ttFont):
        writer.comment('This table will be calculated by the compiler')
        writer.newline()