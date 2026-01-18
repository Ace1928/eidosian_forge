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
@cached_property
def _executing_node_common_indent(self) -> int:
    """
        The common minimal indentation shared by the markers intended
        for an exception node that spans multiple lines.

        Intended to be used only internally.
        """
    indents = []
    lines = [line for line in self.lines if isinstance(line, Line)]
    for line in lines:
        for rang in line._raw_executing_node_ranges():
            begin_text = len(line.text) - len(line.text.lstrip())
            indent = max(rang.start, begin_text)
            indents.append(indent)
    if len(indents) <= 1:
        return 0
    return min(indents[1:])