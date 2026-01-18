from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
def _compare_no_space(self, real_stmt, received_stmt):
    stmt = re.sub('[\\n\\t]', '', real_stmt)
    return received_stmt == stmt