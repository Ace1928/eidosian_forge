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
def _enable_matplotlib_integration():
    """Enable extra IPython matplotlib integration when we are loaded as the matplotlib backend."""
    from matplotlib import get_backend
    ip = get_ipython()
    backend = get_backend()
    if ip and backend == 'module://%s' % __name__:
        from IPython.core.pylabtools import activate_matplotlib
        try:
            activate_matplotlib(backend)
            configure_inline_support(ip, backend)
        except (ImportError, AttributeError):

            def configure_once(*args):
                activate_matplotlib(backend)
                configure_inline_support(ip, backend)
                ip.events.unregister('post_run_cell', configure_once)
            ip.events.register('post_run_cell', configure_once)