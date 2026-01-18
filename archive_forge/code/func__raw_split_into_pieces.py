import ast
import html
import os
import sys
from collections import defaultdict, Counter
from enum import Enum
from textwrap import dedent
from types import FrameType, CodeType, TracebackType
from typing import (
from typing import Mapping
import executing
from asttokens.util import Token
from executing import only
from pure_eval import Evaluator, is_expression_interesting
from stack_data.utils import (
def _raw_split_into_pieces(self, stmt: ast.AST, start: int, end: int) -> Iterator[Tuple[int, int]]:
    for name, body in ast.iter_fields(stmt):
        if isinstance(body, list) and body and isinstance(body[0], (ast.stmt, ast.ExceptHandler, getattr(ast, 'match_case', ()))):
            for rang, group in sorted(group_by_key_func(body, self.line_range).items()):
                sub_stmt = group[0]
                for inner_start, inner_end in self._raw_split_into_pieces(sub_stmt, *rang):
                    if start < inner_start:
                        yield (start, inner_start)
                    if inner_start < inner_end:
                        yield (inner_start, inner_end)
                    start = inner_end
    yield (start, end)