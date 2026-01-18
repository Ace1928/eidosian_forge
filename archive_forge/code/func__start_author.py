import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_author(self, attrs_d):
    self.inauthor = 1
    self.push('author', 1)
    context = self._get_context()
    context.setdefault('authors', [])
    context['authors'].append(FeedParserDict())