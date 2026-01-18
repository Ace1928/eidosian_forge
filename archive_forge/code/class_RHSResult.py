import itertools
import math
from dataclasses import dataclass, field
from typing import (
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import (
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
@dataclass
class RHSResult:
    """Intermediate split result from a right hand split."""
    head: Line
    body: Line
    tail: Line
    opening_bracket: Leaf
    closing_bracket: Leaf