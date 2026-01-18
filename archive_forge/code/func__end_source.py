import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_source(self):
    self.insource = 0
    value = self.pop('source')
    if value:
        self.sourcedata['title'] = value
    self._get_context()['source'] = copy.deepcopy(self.sourcedata)
    self.sourcedata.clear()