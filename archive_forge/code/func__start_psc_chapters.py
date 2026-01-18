import datetime
import re
from .. import util
def _start_psc_chapters(self, attrs_d):
    context = self._get_context()
    if 'psc_chapters' not in context:
        self.psc_chapters_flag = True
        attrs_d['chapters'] = []
        context['psc_chapters'] = util.FeedParserDict(attrs_d)