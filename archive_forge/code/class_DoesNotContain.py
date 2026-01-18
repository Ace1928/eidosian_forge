import operator
from pprint import pformat
import re
import warnings
from ..compat import (
from ..helpers import list_subtract
from ._higherorder import (
from ._impl import (
class DoesNotContain(Mismatch):

    def __init__(self, matchee, needle):
        """Create a DoesNotContain Mismatch.

        :param matchee: the object that did not contain needle.
        :param needle: the needle that 'matchee' was expected to contain.
        """
        self.matchee = matchee
        self.needle = needle

    def describe(self):
        return f'{self.needle!r} not in {self.matchee!r}'