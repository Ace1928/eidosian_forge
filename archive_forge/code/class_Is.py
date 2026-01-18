import operator
from pprint import pformat
import re
import warnings
from ..compat import (
from ..helpers import list_subtract
from ._higherorder import (
from ._impl import (
class Is(_BinaryComparison):
    """Matches if the items are identical."""
    comparator = operator.is_
    mismatch_string = 'is not'