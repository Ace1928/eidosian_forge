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
def _rhs(self: object, line: Line, features: Collection[Feature], mode: Mode) -> Iterator[Line]:
    """Wraps calls to `right_hand_split`.

            The calls increasingly `omit` right-hand trailers (bracket pairs with
            content), meaning the trailers get glued together to split on another
            bracket pair instead.
            """
    for omit in generate_trailers_to_omit(line, mode.line_length):
        lines = list(right_hand_split(line, mode, features, omit=omit))
        if is_line_short_enough(lines[0], mode=mode):
            yield from lines
            return
    yield from right_hand_split(line, mode, features=features)