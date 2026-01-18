from __future__ import generator_stop
from ..util import FeedParserDict
def _gen_georss_coords(value, swap=True, dims=2):
    latlons = (float(ll) for ll in value.replace(',', ' ').split())
    while True:
        try:
            t = [next(latlons), next(latlons)][::swap and -1 or 1]
            if dims == 3:
                t.append(next(latlons))
            yield tuple(t)
        except StopIteration:
            return