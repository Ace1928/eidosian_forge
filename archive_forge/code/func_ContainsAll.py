from ..helpers import map_values
from ._higherorder import (
from ._impl import Mismatch
def ContainsAll(items):
    """Make a matcher that checks whether a list of things is contained
    in another thing.

    The matcher effectively checks that the provided sequence is a subset of
    the matchee.
    """
    from ._basic import Contains
    return MatchesAll(*map(Contains, items), first_only=False)