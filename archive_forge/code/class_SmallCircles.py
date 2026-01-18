import numpy as np
from matplotlib import _api
from matplotlib.path import Path
class SmallCircles(Circles):
    size = 0.2

    def __init__(self, hatch, density):
        self.num_rows = hatch.count('o') * density
        super().__init__(hatch, density)