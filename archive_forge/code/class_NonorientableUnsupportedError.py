from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
class NonorientableUnsupportedError(RuntimeError):

    def __init__(self, mfd):
        RuntimeError.__init__(self, 'Inside view for non-orientable manifolds such as %s is not supported yet.' % mfd.name())