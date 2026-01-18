import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_generator(self):
    value = self.pop('generator')
    context = self._get_context()
    if 'generator_detail' in context:
        context['generator_detail']['name'] = value