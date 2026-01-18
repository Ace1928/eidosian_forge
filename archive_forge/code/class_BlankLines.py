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
class BlankLines(Enum):
    """The values are intended to correspond to the following behaviour:
    HIDDEN: blank lines are not shown in the output
    VISIBLE: blank lines are visible in the output
    SINGLE: any consecutive blank lines are shown as a single blank line
            in the output. This option requires the line number to be shown.
            For a single blank line, the corresponding line number is shown.
            Two or more consecutive blank lines are shown as a single blank
            line in the output with a custom string shown instead of a
            specific line number.
    """
    HIDDEN = 1
    VISIBLE = 2
    SINGLE = 3