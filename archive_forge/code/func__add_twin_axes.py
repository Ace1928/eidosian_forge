from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Bbox
from .mpl_axes import Axes
def _add_twin_axes(self, axes_class, **kwargs):
    """
        Helper for `.twinx`/`.twiny`/`.twin`.

        *kwargs* are forwarded to the parasite axes constructor.
        """
    if axes_class is None:
        axes_class = self._base_axes_class
    ax = parasite_axes_class_factory(axes_class)(self, **kwargs)
    self.parasites.append(ax)
    ax._remove_method = self._remove_any_twin
    return ax