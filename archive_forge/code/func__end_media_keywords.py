from ..util import FeedParserDict
def _end_media_keywords(self):
    for term in self.pop('media_keywords').split(','):
        if term.strip():
            self._add_tag(term.strip(), None, None)