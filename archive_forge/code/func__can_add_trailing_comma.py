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
def _can_add_trailing_comma(leaf: Leaf, features: Collection[Feature]) -> bool:
    if is_vararg(leaf, within={syms.typedargslist}):
        return Feature.TRAILING_COMMA_IN_DEF in features
    if is_vararg(leaf, within={syms.arglist, syms.argument}):
        return Feature.TRAILING_COMMA_IN_CALL in features
    return True