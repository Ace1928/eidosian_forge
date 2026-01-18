import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_summary(self, attrs_d):
    context = self._get_context()
    if 'summary' in context and (not self.hasContent):
        self._summaryKey = 'content'
        self._start_content(attrs_d)
    else:
        self._summaryKey = 'summary'
        self.push_content(self._summaryKey, attrs_d, 'text/plain', 1)