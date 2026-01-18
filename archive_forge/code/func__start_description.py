import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_description(self, attrs_d):
    context = self._get_context()
    if 'summary' in context and (not self.hasContent):
        self._summaryKey = 'content'
        self._start_content(attrs_d)
    else:
        self.push_content('description', attrs_d, 'text/html', self.infeed or self.inentry or self.insource)