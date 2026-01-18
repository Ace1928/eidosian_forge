import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_content(self):
    copyToSummary = self.map_content_type(self.contentparams.get('type')) in {'text/plain'} | self.html_types
    value = self.pop_content('content')
    if copyToSummary:
        self._save('summary', value)