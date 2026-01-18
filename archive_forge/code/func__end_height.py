import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_height(self):
    value = self.pop('height')
    try:
        value = int(value)
    except ValueError:
        value = 0
    if self.inimage:
        context = self._get_context()
        context['height'] = value