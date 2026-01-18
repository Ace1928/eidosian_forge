import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_image(self, attrs_d):
    context = self._get_context()
    if not self.inentry:
        context.setdefault('image', FeedParserDict())
    self.inimage = 1
    self.title_depth = -1
    self.push('image', 0)