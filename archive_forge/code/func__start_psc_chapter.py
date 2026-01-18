import datetime
import re
from .. import util
def _start_psc_chapter(self, attrs_d):
    if self.psc_chapters_flag:
        start = self._get_attribute(attrs_d, 'start')
        attrs_d['start_parsed'] = _parse_psc_chapter_start(start)
        context = self._get_context()['psc_chapters']
        context['chapters'].append(util.FeedParserDict(attrs_d))