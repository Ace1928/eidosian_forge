from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
def _compare_dicts(self, expected, observed):
    common_keys = set(expected.keys()) & set(observed.keys())
    mismatches = {}
    for key in common_keys:
        mismatch = expected[key].match(observed[key])
        if mismatch:
            mismatches[key] = mismatch
    return mismatches