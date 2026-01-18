import flatbuffers
from flatbuffers.compat import import_numpy
class BucketizeOptionsT(object):

    def __init__(self):
        self.boundaries = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        bucketizeOptions = BucketizeOptions()
        bucketizeOptions.Init(buf, pos)
        return cls.InitFromObj(bucketizeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, bucketizeOptions):
        x = BucketizeOptionsT()
        x._UnPack(bucketizeOptions)
        return x

    def _UnPack(self, bucketizeOptions):
        if bucketizeOptions is None:
            return
        if not bucketizeOptions.BoundariesIsNone():
            if np is None:
                self.boundaries = []
                for i in range(bucketizeOptions.BoundariesLength()):
                    self.boundaries.append(bucketizeOptions.Boundaries(i))
            else:
                self.boundaries = bucketizeOptions.BoundariesAsNumpy()

    def Pack(self, builder):
        if self.boundaries is not None:
            if np is not None and type(self.boundaries) is np.ndarray:
                boundaries = builder.CreateNumpyVector(self.boundaries)
            else:
                BucketizeOptionsStartBoundariesVector(builder, len(self.boundaries))
                for i in reversed(range(len(self.boundaries))):
                    builder.PrependFloat32(self.boundaries[i])
                boundaries = builder.EndVector()
        BucketizeOptionsStart(builder)
        if self.boundaries is not None:
            BucketizeOptionsAddBoundaries(builder, boundaries)
        bucketizeOptions = BucketizeOptionsEnd(builder)
        return bucketizeOptions