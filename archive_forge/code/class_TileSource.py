from __future__ import annotations
import logging # isort:skip
from ..core.properties import (
from ..model import Model
class TileSource(Model):
    """ A base class for all tile source types.

    In general, tile sources are used as a required input for ``TileRenderer``.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    _args = ('url', 'tile_size', 'min_zoom', 'max_zoom', 'extra_url_vars')
    url = String('', help='\n    Tile service url e.g., http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png\n    ')
    tile_size = Int(default=256, help='\n    Tile size in pixels (e.g. 256)\n    ')
    min_zoom = Int(default=0, help='\n    A minimum zoom level for the tile layer. This is the most zoomed-out level.\n    ')
    max_zoom = Int(default=30, help='\n    A maximum zoom level for the tile layer. This is the most zoomed-in level.\n    ')
    extra_url_vars = Dict(String, Any, help='\n    A dictionary that maps url variable template keys to values.\n\n    These variables are useful for parts of tile urls which do not change from\n    tile to tile (e.g. server host name, or layer name).\n    ')
    attribution = String('', help='\n    Data provider attribution content. This can include HTML content.\n    ')
    x_origin_offset = Required(Float, help='\n    An x-offset in plot coordinates\n    ')
    y_origin_offset = Required(Float, help='\n    A y-offset in plot coordinates\n    ')
    initial_resolution = Nullable(Float, help='\n    Resolution (plot_units / pixels) of minimum zoom level of tileset\n    projection. None to auto-compute.\n    ')