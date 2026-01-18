import numpy as np
from matplotlib import _api
from matplotlib.path import Path
class Circles(Shapes):

    def __init__(self, hatch, density):
        path = Path.unit_circle()
        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        super().__init__(hatch, density)