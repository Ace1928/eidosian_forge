from __future__ import generator_stop
from ..util import FeedParserDict
def _start_gml_polygon(self, attrs_d):
    self._parse_srs_attrs(attrs_d)
    self.push('geometry', 0)