from __future__ import generator_stop
from ..util import FeedParserDict
def _parse_georss_box(value, swap=True, dims=2):
    try:
        coords = list(_gen_georss_coords(value, swap, dims))
        return {'type': 'Box', 'coordinates': tuple(coords)}
    except (IndexError, ValueError):
        return None