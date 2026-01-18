import numpy as np
from matplotlib import _api
from matplotlib.path import Path
class LargeCircles(Circles):
    size = 0.35

    def __init__(self, hatch, density):
        self.num_rows = hatch.count('O') * density
        super().__init__(hatch, density)