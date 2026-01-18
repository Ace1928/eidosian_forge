from ..util import FeedParserDict
def _start_itunes_image(self, attrs_d):
    self.push('itunes_image', 0)
    if attrs_d.get('href'):
        self._get_context()['image'] = FeedParserDict({'href': attrs_d.get('href')})
    elif attrs_d.get('url'):
        self._get_context()['image'] = FeedParserDict({'href': attrs_d.get('url')})