import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _hugging_power_ops_line_to_string(line: Line, features: Collection[Feature], mode: Mode) -> Optional[str]:
    try:
        return line_to_string(next(hug_power_op(line, features, mode)))
    except CannotTransform:
        return None