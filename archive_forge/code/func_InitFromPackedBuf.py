import flatbuffers
from flatbuffers.compat import import_numpy
@classmethod
def InitFromPackedBuf(cls, buf, pos=0):
    n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
    return cls.InitFromBuf(buf, pos + n)