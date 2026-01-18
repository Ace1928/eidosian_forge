import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_contributor(self, attrs_d):
    self.incontributor = 1
    context = self._get_context()
    context.setdefault('contributors', [])
    context['contributors'].append(FeedParserDict())
    self.push('contributor', 0)