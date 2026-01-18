import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_body(self, attrs_d):
    self.push_content('content', attrs_d, 'application/xhtml+xml', 1)