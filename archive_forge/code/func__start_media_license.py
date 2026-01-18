from ..util import FeedParserDict
def _start_media_license(self, attrs_d):
    context = self._get_context()
    context.setdefault('media_license', attrs_d)
    self.push('license', 1)