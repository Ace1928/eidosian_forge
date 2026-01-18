from __future__ import annotations
import sys
from . import config
from . import exclusions
from .. import event
from .. import schema
from .. import types as sqltypes
from ..orm import mapped_column as _orm_mapped_column
from ..util import OrderedDict
class eq_clause_element:
    """Helper to compare SQL structures based on compare()"""

    def __init__(self, target):
        self.target = target

    def __eq__(self, other):
        return self.target.compare(other)

    def __ne__(self, other):
        return not self.target.compare(other)