import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_newlocation(self):
    url = self.pop('newlocation')
    context = self._get_context()
    if context is not self.feeddata:
        return
    context['newlocation'] = make_safe_absolute_uri(self.baseuri, url.strip())