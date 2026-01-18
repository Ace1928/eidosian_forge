from __future__ import absolute_import
from __future__ import print_function
import webbrowser
from ..palette import Palette
from .colorbrewer_all_schemes import COLOR_MAPS
def _load_maps_by_type(map_type):
    """
    Load all maps of a given type into a dictionary.

    Color maps are loaded as BrewerMap objects. Dictionary is
    keyed by map name and then integer numbers of defined
    colors. There is an additional 'max' key that points to the
    color map with the largest number of defined colors.

    Parameters
    ----------
    map_type : {'sequential', 'diverging', 'qualitative'}

    Returns
    -------
    maps : dict of BrewerMap

    """
    map_type = map_type.capitalize()
    seq_maps = COLOR_MAPS[map_type]
    loaded_maps = {}
    for map_name in seq_maps:
        for num in seq_maps[map_name]:
            inum = int(num)
            name = '{0}_{1}'.format(map_name, num)
            loaded_maps[name] = get_map(map_name, map_type, inum)
            loaded_maps[name + '_r'] = get_map(map_name, map_type, inum, reverse=True)
    return loaded_maps