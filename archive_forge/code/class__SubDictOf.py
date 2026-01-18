from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class _SubDictOf(Matcher):
    """Matches if the matched dict only has keys that are in given dict."""

    def __init__(self, super_dict, format_value=repr):
        super().__init__()
        self.super_dict = super_dict
        self.format_value = format_value

    def match(self, observed):
        excess = dict_subtract(observed, self.super_dict)
        return _dict_to_mismatch(excess, lambda v: Mismatch(self.format_value(v)))