from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class ContainedByDict(_CombinedMatcher):
    """Match a dictionary for which this is a super-dictionary.

    Specify a dictionary mapping keys (often strings) to matchers.  This is
    the 'expected' dict.  Any dictionary that matches this must have **only**
    these keys, and the values must match the corresponding matchers in the
    expected dict.  Dictionaries that have fewer keys can also match.

    In other words, any matching dictionary must be contained by the
    dictionary given to the constructor.

    Does not check for strict super-dictionary.  That is, equal dictionaries
    match.
    """
    matcher_factories = {'Extra': _SubDictOf, 'Differences': _MatchCommonKeys}
    format_expected = lambda self, expected: _format_matcher_dict(expected)