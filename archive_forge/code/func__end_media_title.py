from ..util import FeedParserDict
def _end_media_title(self):
    title_depth = self.title_depth
    self._end_title()
    self.title_depth = title_depth