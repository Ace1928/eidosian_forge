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
class SkewXTick(maxis.XTick):

    def draw(self, renderer):
        with ExitStack() as stack:
            for artist in [self.gridline, self.tick1line, self.tick2line, self.label1, self.label2]:
                stack.callback(artist.set_visible, artist.get_visible())
            needs_lower = transforms.interval_contains(self.axes.lower_xlim, self.get_loc())
            needs_upper = transforms.interval_contains(self.axes.upper_xlim, self.get_loc())
            self.tick1line.set_visible(self.tick1line.get_visible() and needs_lower)
            self.label1.set_visible(self.label1.get_visible() and needs_lower)
            self.tick2line.set_visible(self.tick2line.get_visible() and needs_upper)
            self.label2.set_visible(self.label2.get_visible() and needs_upper)
            super().draw(renderer)

    def get_view_interval(self):
        return self.axes.xaxis.get_view_interval()