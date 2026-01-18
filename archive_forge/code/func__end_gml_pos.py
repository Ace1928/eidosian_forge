from __future__ import generator_stop
from ..util import FeedParserDict
def _end_gml_pos(self):
    this = self.pop('pos')
    context = self._get_context()
    srs_name = context['where'].get('srsName')
    srs_dimension = context['where'].get('srsDimension', 2)
    swap = True
    if srs_name and 'EPSG' in srs_name:
        epsg = int(srs_name.split(':')[-1])
        swap = bool(epsg in _geogCS)
    geometry = _parse_georss_point(this, swap=swap, dims=srs_dimension)
    if geometry:
        self._save_where(geometry)