import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_updated(self):
    value = self.pop('updated')
    parsed_value = _parse_date(value)
    self._save('updated_parsed', parsed_value, overwrite=True)