import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_channel(self, attrs_d):
    self.infeed = 1
    self._cdf_common(attrs_d)