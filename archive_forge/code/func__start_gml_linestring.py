from __future__ import generator_stop
from ..util import FeedParserDict
def _start_gml_linestring(self, attrs_d):
    self._parse_srs_attrs(attrs_d)
    self.ingeometry = 'linestring'
    self.push('geometry', 0)