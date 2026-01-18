import numpy as np
from matplotlib import _api
from matplotlib.path import Path
class SouthEastHatch(HatchPatternBase):

    def __init__(self, hatch, density):
        self.num_lines = int((hatch.count('\\') + hatch.count('x') + hatch.count('X')) * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 1.0 + steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 0.0 + steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO