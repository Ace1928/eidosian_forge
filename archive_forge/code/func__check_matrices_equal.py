from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def _check_matrices_equal(m1, m2):
    for i in range(4):
        for j in range(4):
            if abs(m1[i][j] - m2[i][j]) > 1e-10:
                print(m1, m2)
                print('Matrix not zero as expected')
                return