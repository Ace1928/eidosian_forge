import wx
from .backend_agg import FigureCanvasAgg
from .backend_wx import _BackendWx, _FigureCanvasWxBase
from .backend_wx import (  # noqa: F401 # pylint: disable=W0611
def _rgba_to_wx_bitmap(rgba):
    """Convert an RGBA buffer to a wx.Bitmap."""
    h, w, _ = rgba.shape
    return wx.Bitmap.FromBufferRGBA(w, h, rgba)