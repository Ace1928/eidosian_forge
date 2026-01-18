from io import BytesIO
import functools
from fontTools import subset
import matplotlib as mpl
from .. import font_manager, ft2font
from .._afm import AFM
from ..backend_bases import RendererBase
def _get_font_afm(self, prop):
    fname = font_manager.findfont(prop, fontext='afm', directory=self._afm_font_dir)
    return _cached_get_afm_from_fname(fname)