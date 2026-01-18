import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_summary(self):
    if self._summaryKey == 'content':
        self._end_content()
    else:
        self.pop_content(self._summaryKey or 'summary')
    self._summaryKey = None