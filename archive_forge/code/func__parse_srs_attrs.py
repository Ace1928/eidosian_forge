from __future__ import generator_stop
from ..util import FeedParserDict
def _parse_srs_attrs(self, attrs_d):
    srs_name = attrs_d.get('srsname')
    try:
        srs_dimension = int(attrs_d.get('srsdimension', '2'))
    except ValueError:
        srs_dimension = 2
    context = self._get_context()
    if 'where' not in context:
        context['where'] = {}
    context['where']['srsName'] = srs_name
    context['where']['srsDimension'] = srs_dimension