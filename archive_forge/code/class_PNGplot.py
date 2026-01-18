from rpy2 import robjects
from rpy2.robjects.lib import ggplot2, grdevices
from IPython import get_ipython  # type: ignore
from IPython.core.display import Image  # type: ignore
class PNGplot(object):
    """
    Context manager
    """

    def __init__(self, width=600, height=400):
        self._width = width
        self._height = height
        png_formatter = get_ipython().display_formatter.formatters['image/png']
        self._png_formatter = png_formatter
        self._for_ggplot = self._png_formatter.for_type(ggplot2.GGPlot)

    def __enter__(self):
        self._png_formatter.for_type(ggplot2.GGPlot, display_png)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._png_formatter.for_type(ggplot2.GGPlot, self._for_ggplot)
        return False