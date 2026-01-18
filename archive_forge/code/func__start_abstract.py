import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_abstract(self, attrs_d):
    self.push_content('description', attrs_d, 'text/plain', self.infeed or self.inentry or self.insource)