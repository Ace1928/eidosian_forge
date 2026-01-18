import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_enclosure(self, attrs_d):
    attrs_d = self._enforce_href(attrs_d)
    context = self._get_context()
    attrs_d['rel'] = 'enclosure'
    context.setdefault('links', []).append(FeedParserDict(attrs_d))