from ..util import FeedParserDict
def _start_media_rating(self, attrs_d):
    context = self._get_context()
    context.setdefault('media_rating', attrs_d)
    self.push('rating', 1)