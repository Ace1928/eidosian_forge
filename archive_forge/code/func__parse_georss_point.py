from __future__ import generator_stop
from ..util import FeedParserDict
def _parse_georss_point(value, swap=True, dims=2):
    try:
        coords = list(_gen_georss_coords(value, swap, dims))
        return {'type': 'Point', 'coordinates': coords[0]}
    except (IndexError, ValueError):
        return None