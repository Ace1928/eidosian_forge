from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
class MissingOperatorWithDecomp(OperatorIssue):

    def __init__(self, target, args, kwargs):
        _record_missing_op(target)
        super().__init__(f'missing decomposition\n{self.operator_str(target, args, kwargs)}' + textwrap.dedent(f'\n\n                There is a decomposition available for {target} in\n                torch._decomp.get_decompositions().  Please add this operator to the\n                `decompositions` list in torch._inductor.decompositions\n                '))