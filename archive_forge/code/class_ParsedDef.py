import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, List, NamedTuple, Optional, Tuple
from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory
class ParsedDef(NamedTuple):
    ast: ast.Module
    ctx: SourceContext
    source: str
    filename: Optional[str]
    file_lineno: int