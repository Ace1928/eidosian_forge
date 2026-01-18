from ..util import FeedParserDict
def _end_media_restriction(self):
    restriction = self.pop('restriction')
    if restriction is not None and restriction.strip():
        context = self._get_context()
        context['media_restriction']['content'] = [cc.strip().lower() for cc in restriction.split(' ')]