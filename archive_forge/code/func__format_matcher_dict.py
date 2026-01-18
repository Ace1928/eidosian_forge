from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
def _format_matcher_dict(matchers):
    return '{%s}' % ', '.join(sorted((f'{k!r}: {v}' for k, v in matchers.items())))