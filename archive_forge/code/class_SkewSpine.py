from contextlib import ExitStack
import itertools
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
import matplotlib.patches as mpatch
from matplotlib.projections import register_projection
class SkewSpine(mspines.Spine):

    def _adjust_location(self):
        pts = self._path.vertices
        if self.spine_type == 'top':
            pts[:, 0] = self.axes.upper_xlim
        else:
            pts[:, 0] = self.axes.lower_xlim