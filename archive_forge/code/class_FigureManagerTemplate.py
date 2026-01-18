from matplotlib import _api
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.figure import Figure
class FigureManagerTemplate(FigureManagerBase):
    """
    Helper class for pyplot mode, wraps everything up into a neat bundle.

    For non-interactive backends, the base class is sufficient.  For
    interactive backends, see the documentation of the `.FigureManagerBase`
    class for the list of methods that can/should be overridden.
    """