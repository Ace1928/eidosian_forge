from __future__ import absolute_import, print_function
from ..palette import Palette
class TableauMap(Palette):
    """
    Representation of a color map with matplotlib compatible
    views of the map.

    Parameters
    ----------
    name : str
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

    """
    url = url

    def __init__(self, name, colors):
        super(TableauMap, self).__init__(name, palette_type, colors)