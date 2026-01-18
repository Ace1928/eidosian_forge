from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Bbox
from .mpl_axes import Axes
def _remove_any_twin(self, ax):
    self.parasites.remove(ax)
    restore = ['top', 'right']
    if ax._sharex:
        restore.remove('top')
    if ax._sharey:
        restore.remove('right')
    self.axis[tuple(restore)].set_visible(True)
    self.axis[tuple(restore)].toggle(ticklabels=False, label=False)