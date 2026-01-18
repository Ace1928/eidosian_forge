import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_category(self, attrs_d):
    term = attrs_d.get('term')
    scheme = attrs_d.get('scheme', attrs_d.get('domain'))
    label = attrs_d.get('label')
    self._add_tag(term, scheme, label)
    self.push('category', 1)