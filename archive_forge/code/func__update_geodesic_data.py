from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def _update_geodesic_data(self):
    success = self.geodesics.set_enables_and_radii_and_update(self.ui_parameter_dict['geodesicTubeEnables'][1], self.ui_parameter_dict['geodesicTubeRadii'][1])
    self.geodesics_uniform_bindings = self.geodesics.get_uniform_bindings()
    return success