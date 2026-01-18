import matplotlib
from matplotlib import colors
from matplotlib.backends import backend_agg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.getipython import get_ipython
from IPython.core.pylabtools import select_figure_formats
from IPython.display import display
from .config import InlineBackend
def _fetch_figure_metadata(fig):
    """Get some metadata to help with displaying a figure."""
    if _is_transparent(fig.get_facecolor()):
        ticksLight = _is_light([label.get_color() for axes in fig.axes for axis in (axes.xaxis, axes.yaxis) for label in axis.get_ticklabels()])
        if ticksLight.size and (ticksLight == ticksLight[0]).all():
            return {'needs_background': 'dark' if ticksLight[0] else 'light'}
    return None