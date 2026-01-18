from __future__ import generator_stop
from ..util import FeedParserDict
def _end_georss_polygon(self):
    this = self.pop('geometry')
    geometry = _parse_georss_polygon(this)
    if geometry:
        self._save_where(geometry)