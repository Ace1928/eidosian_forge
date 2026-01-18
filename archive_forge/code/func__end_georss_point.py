from __future__ import generator_stop
from ..util import FeedParserDict
def _end_georss_point(self):
    geometry = _parse_georss_point(self.pop('geometry'))
    if geometry:
        self._save_where(geometry)