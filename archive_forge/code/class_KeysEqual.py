from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class KeysEqual(Matcher):
    """Checks whether a dict has particular keys."""

    def __init__(self, *expected):
        """Create a `KeysEqual` Matcher.

        :param expected: The keys the matchee is expected to have. As a
            special case, if a single argument is specified, and it is a
            mapping, then we use its keys as the expected set.
        """
        super().__init__()
        if len(expected) == 1:
            try:
                expected = expected[0].keys()
            except AttributeError:
                pass
        self.expected = list(expected)

    def __str__(self):
        return 'KeysEqual(%s)' % ', '.join(map(repr, self.expected))

    def match(self, matchee):
        from ._basic import _BinaryMismatch, Equals
        expected = sorted(self.expected)
        matched = Equals(expected).match(sorted(matchee.keys()))
        if matched:
            return AnnotatedMismatch('Keys not equal', _BinaryMismatch(expected, 'does not match', matchee))
        return None