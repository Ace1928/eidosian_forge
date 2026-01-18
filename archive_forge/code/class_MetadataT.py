import flatbuffers
from flatbuffers.compat import import_numpy
class MetadataT(object):

    def __init__(self):
        self.name = None
        self.buffer = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        metadata = Metadata()
        metadata.Init(buf, pos)
        return cls.InitFromObj(metadata)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, metadata):
        x = MetadataT()
        x._UnPack(metadata)
        return x

    def _UnPack(self, metadata):
        if metadata is None:
            return
        self.name = metadata.Name()
        self.buffer = metadata.Buffer()

    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        MetadataStart(builder)
        if self.name is not None:
            MetadataAddName(builder, name)
        MetadataAddBuffer(builder, self.buffer)
        metadata = MetadataEnd(builder)
        return metadata