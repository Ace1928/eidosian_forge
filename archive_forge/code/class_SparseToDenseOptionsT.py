import flatbuffers
from flatbuffers.compat import import_numpy
class SparseToDenseOptionsT(object):

    def __init__(self):
        self.validateIndices = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sparseToDenseOptions = SparseToDenseOptions()
        sparseToDenseOptions.Init(buf, pos)
        return cls.InitFromObj(sparseToDenseOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, sparseToDenseOptions):
        x = SparseToDenseOptionsT()
        x._UnPack(sparseToDenseOptions)
        return x

    def _UnPack(self, sparseToDenseOptions):
        if sparseToDenseOptions is None:
            return
        self.validateIndices = sparseToDenseOptions.ValidateIndices()

    def Pack(self, builder):
        SparseToDenseOptionsStart(builder)
        SparseToDenseOptionsAddValidateIndices(builder, self.validateIndices)
        sparseToDenseOptions = SparseToDenseOptionsEnd(builder)
        return sparseToDenseOptions