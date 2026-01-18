import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_rss(self, attrs_d):
    versionmap = {'0.91': 'rss091u', '0.92': 'rss092', '0.93': 'rss093', '0.94': 'rss094'}
    if not self.version or not self.version.startswith('rss'):
        attr_version = attrs_d.get('version', '')
        version = versionmap.get(attr_version)
        if version:
            self.version = version
        elif attr_version.startswith('2.'):
            self.version = 'rss20'
        else:
            self.version = 'rss'