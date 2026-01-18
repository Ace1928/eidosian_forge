import ctypes, logging
from OpenGL import platform, error
from OpenGL._configflags import STORE_POINTERS, ERROR_ON_COPY, SIZE_1_ARRAY_UNPACK
from OpenGL import converters
from OpenGL.converters import DefaultCConverter
from OpenGL.converters import returnCArgument,returnPyArgument
from OpenGL.latebind import LateBind
from OpenGL.arrays import arrayhelpers, arraydatatype
from OpenGL._null import NULL
from OpenGL import acceleratesupport
class MultiReturn(object):

    def __init__(self, *children):
        self.children = list(children)

    def append(self, child):
        self.children.append(child)

    def __call__(self, *args, **named):
        result = []
        for child in self.children:
            try:
                result.append(child(*args, **named))
            except Exception as err:
                err.args += (child, args, named)
                raise
        return result