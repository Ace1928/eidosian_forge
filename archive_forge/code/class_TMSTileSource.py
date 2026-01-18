from __future__ import annotations
import logging # isort:skip
from ..core.properties import (
from ..model import Model
class TMSTileSource(MercatorTileSource):
    """ Contains tile config info and provides urls for tiles based on a
    templated url e.g. ``http://your.tms.server.host/{Z}/{X}/{Y}.png``. The
    defining feature of TMS is the tile-origin in located at the bottom-left.

    ``TMSTileSource`` can also be helpful in implementing tile renderers for
    custom tile sets, including non-spatial datasets.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)