from ..util import FeedParserDict
def _end_media_thumbnail(self):
    url = self.pop('url')
    context = self._get_context()
    if url is not None and url.strip():
        if 'url' not in context['media_thumbnail'][-1]:
            context['media_thumbnail'][-1]['url'] = url