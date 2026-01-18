from __future__ import generator_stop
from ..util import FeedParserDict
def _save_where(self, geometry):
    context = self._get_context()
    context['where'].update(geometry)