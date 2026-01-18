import doctest
import re
from ._impl import Mismatch
def _with_nl(self, actual):
    result = self.want.__class__(actual)
    if not result.endswith('\n'):
        result += '\n'
    return result