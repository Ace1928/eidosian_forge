from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
class LoweringException(OperatorIssue):

    def __init__(self, exc: Exception, target, args, kwargs):
        super().__init__(f'{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}')