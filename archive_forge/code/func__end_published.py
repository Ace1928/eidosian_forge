import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_published(self):
    value = self.pop('published')
    self._save('published_parsed', _parse_date(value), overwrite=True)