from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
def _dict_to_mismatch(data, to_mismatch=None, result_mismatch=DictMismatches):
    if to_mismatch:
        data = map_values(to_mismatch, data)
    mismatches = filter_values(bool, data)
    if mismatches:
        return result_mismatch(mismatches)