from io import BytesIO
from binascii import b2a_base64
from functools import partial
import warnings
from IPython.core.display import _pngxy
from IPython.utils.decorators import flag_calls
def activate_matplotlib(backend):
    """Activate the given backend and set interactive to True."""
    import matplotlib
    matplotlib.interactive(True)
    matplotlib.rcParams['backend'] = backend
    from matplotlib import pyplot as plt
    plt.switch_backend(backend)
    plt.show._needmain = False
    plt.draw_if_interactive = flag_calls(plt.draw_if_interactive)