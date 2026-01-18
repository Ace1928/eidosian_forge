import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_name(self):
    value = self.pop('name')
    if self.inpublisher:
        self._save_author('name', value, 'publisher')
    elif self.inauthor:
        self._save_author('name', value)
    elif self.incontributor:
        self._save_contributor('name', value)
    elif self.intextinput:
        context = self._get_context()
        context['name'] = value