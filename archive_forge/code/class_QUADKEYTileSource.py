from __future__ import annotations
import logging # isort:skip
from ..core.properties import (
from ..model import Model
class QUADKEYTileSource(MercatorTileSource):
    """ Has the same tile origin as the ``WMTSTileSource`` but requests tiles using
    a `quadkey` argument instead of X, Y, Z e.g.
    ``http://your.quadkey.tile.host/{Q}.png``

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)