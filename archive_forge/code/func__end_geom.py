from __future__ import generator_stop
from ..util import FeedParserDict
def _end_geom(self):
    self.ingeometry = 0
    self.pop('geometry')