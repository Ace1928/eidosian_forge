from ..datetimes import _parse_date
from ..util import FeedParserDict
def _start_dc_contributor(self, attrs_d):
    self.incontributor = 1
    context = self._get_context()
    context.setdefault('contributors', [])
    context['contributors'].append(FeedParserDict())
    self.push('name', 0)