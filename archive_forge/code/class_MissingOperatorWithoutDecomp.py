from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
class MissingOperatorWithoutDecomp(OperatorIssue):

    def __init__(self, target, args, kwargs):
        _record_missing_op(target)
        super().__init__(f'missing lowering\n{self.operator_str(target, args, kwargs)}')