from contextlib import nullcontext
from math import radians, cos, sin
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.backend_bases import (
from matplotlib.font_manager import fontManager as _fontManager, get_font
from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
def _update_methods(self):
    self.draw_gouraud_triangle = self._renderer.draw_gouraud_triangle
    self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
    self.draw_image = self._renderer.draw_image
    self.draw_markers = self._renderer.draw_markers
    self.draw_path_collection = self._renderer.draw_path_collection
    self.draw_quad_mesh = self._renderer.draw_quad_mesh
    self.copy_from_bbox = self._renderer.copy_from_bbox