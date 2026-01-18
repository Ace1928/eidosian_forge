from __future__ import generator_stop
from ..util import FeedParserDict
def _start_gml_pos(self, attrs_d):
    self.push('pos', 0)