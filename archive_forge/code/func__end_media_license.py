from ..util import FeedParserDict
def _end_media_license(self):
    license_ = self.pop('license')
    if license_ is not None and license_.strip():
        context = self._get_context()
        context['media_license']['content'] = license_