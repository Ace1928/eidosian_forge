import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_cloud(self, attrs_d):
    self._get_context()['cloud'] = FeedParserDict(attrs_d)