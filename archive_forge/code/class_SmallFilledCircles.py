import numpy as np
from matplotlib import _api
from matplotlib.path import Path
class SmallFilledCircles(Circles):
    size = 0.1
    filled = True

    def __init__(self, hatch, density):
        self.num_rows = hatch.count('.') * density
        super().__init__(hatch, density)