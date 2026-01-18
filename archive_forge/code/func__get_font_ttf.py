from io import BytesIO
import functools
from fontTools import subset
import matplotlib as mpl
from .. import font_manager, ft2font
from .._afm import AFM
from ..backend_bases import RendererBase
def _get_font_ttf(self, prop):
    fnames = font_manager.fontManager._find_fonts_by_props(prop)
    font = font_manager.get_font(fnames)
    font.clear()
    font.set_size(prop.get_size_in_points(), 72)
    return font