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
class LineGap(object):
    """
    A singleton representing one or more lines of source code that were skipped
    in FrameInfo.lines.

    LINE_GAP can be created in two ways:
    - by truncating a piece of context that's too long.
    - immediately after the signature piece if Options.include_signature is true
      and the following piece isn't already part of the included pieces. 
    """

    def __repr__(self):
        return 'LINE_GAP'