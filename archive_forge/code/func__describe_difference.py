import doctest
import re
from ._impl import Mismatch
def _describe_difference(self, with_nl):
    return self._checker.output_difference(self, with_nl, self.flags)