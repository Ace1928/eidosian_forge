from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def _update_shader(self):
    if self.geodesics:
        geodesic_compile_time_constants = self.geodesics.get_compile_time_constants()
    else:
        geodesic_compile_time_constants = {b'##num_geodesic_segments##': 0}
    compile_time_constants = _merge_dicts(self.raytracing_data.get_compile_time_constants(), geodesic_compile_time_constants)
    if compile_time_constants == self.compile_time_constants:
        return
    self.compile_time_constants = compile_time_constants
    shader_source, uniform_block_names_sizes_and_offsets = shaders.get_triangulation_shader_source_and_ubo_descriptors(compile_time_constants)
    self.set_fragment_shader_source(shader_source, uniform_block_names_sizes_and_offsets)