import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_email(self):
    value = self.pop('email')
    if self.inpublisher:
        self._save_author('email', value, 'publisher')
    elif self.inauthor:
        self._save_author('email', value)
    elif self.incontributor:
        self._save_contributor('email', value)