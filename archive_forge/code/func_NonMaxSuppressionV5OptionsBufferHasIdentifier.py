import flatbuffers
from flatbuffers.compat import import_numpy
@classmethod
def NonMaxSuppressionV5OptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
    return flatbuffers.util.BufferHasIdentifier(buf, offset, b'TFL3', size_prefixed=size_prefixed)