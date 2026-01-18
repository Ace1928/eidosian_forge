import flatbuffers
from flatbuffers.compat import import_numpy
@classmethod
def GetRootAs(cls, buf, offset=0):
    n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
    x = Model()
    x.Init(buf, n + offset)
    return x