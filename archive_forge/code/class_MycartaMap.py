from __future__ import absolute_import, print_function
import itertools
from . import colordata
from .. import utils
from ..palette import Palette
class MycartaMap(Palette):
    """
    Representation of a color map with matplotlib compatible
    views of the map.

    Parameters
    ----------
    name : str
    palette_type : str
    colors : list
        Colors as list of 0-255 RGB triplets.

    Attributes
    ----------
    name : str
    type : str
    number : int
        Number of colors in color map.
    colors : list
        Colors as list of 0-255 RGB triplets.
    hex_colors : list
    mpl_colors : list
    mpl_colormap : matplotlib LinearSegmentedColormap
    url : str
        Website with related info.

    """
    url = 'https://mycarta.wordpress.com/color-palettes/'

    def __init__(self, name, palette_type, colors):
        super(MycartaMap, self).__init__(name, palette_type, colors)