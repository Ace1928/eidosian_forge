import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_guid(self, attrs_d):
    self.guidislink = attrs_d.get('ispermalink', 'true') == 'true'
    self.push('id', 1)