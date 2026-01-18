from ..util import FeedParserDict
def _end_media_rating(self):
    rating = self.pop('rating')
    if rating is not None and rating.strip():
        context = self._get_context()
        context['media_rating']['content'] = rating