from io import BytesIO
from binascii import b2a_base64
from functools import partial
import warnings
from IPython.core.display import _pngxy
from IPython.utils.decorators import flag_calls
def _reshow_nbagg_figure(fig):
    """reshow an nbagg figure"""
    try:
        reshow = fig.canvas.manager.reshow
    except AttributeError as e:
        raise NotImplementedError() from e
    else:
        reshow()