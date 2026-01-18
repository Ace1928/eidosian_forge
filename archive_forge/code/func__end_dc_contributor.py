from ..datetimes import _parse_date
from ..util import FeedParserDict
def _end_dc_contributor(self):
    self._end_name()
    self.incontributor = 0