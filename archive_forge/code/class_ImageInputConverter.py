from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
class ImageInputConverter(object):

    def __init__(self, rank, pixelsName=None, typeName='type'):
        self.rank = rank
        self.typeName = typeName
        self.pixelsName = pixelsName

    def finalise(self, wrapper):
        """Get our pixel index from the wrapper"""
        self.typeIndex = wrapper.pyArgIndex(self.typeName)
        self.pixelsIndex = wrapper.pyArgIndex(self.pixelsName)

    def __call__(self, arg, baseOperation, pyArgs):
        """pyConverter for the pixels argument"""
        images.setupDefaultTransferMode()
        images.rankPacking(self.rank)
        type = pyArgs[self.typeIndex]
        arrayType = arrays.GL_CONSTANT_TO_ARRAY_TYPE[images.TYPE_TO_ARRAYTYPE[type]]
        return arrayType.asArray(arg)