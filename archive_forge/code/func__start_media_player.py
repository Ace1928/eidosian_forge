from ..util import FeedParserDict
def _start_media_player(self, attrs_d):
    self.push('media_player', 0)
    self._get_context()['media_player'] = FeedParserDict(attrs_d)