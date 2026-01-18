import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_title(self, attrs_d):
    if self.svgOK:
        return self.unknown_starttag('title', list(attrs_d.items()))
    self.push_content('title', attrs_d, 'text/plain', self.infeed or self.inentry or self.insource)