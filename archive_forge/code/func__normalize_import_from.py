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
def _normalize_import_from(parent: Node, child: LN, index: int) -> None:
    if is_lpar_token(child):
        assert is_rpar_token(parent.children[-1])
        child.value = ''
        parent.children[-1].value = ''
    elif child.type != token.STAR:
        parent.insert_child(index, Leaf(token.LPAR, ''))
        parent.append_child(Leaf(token.RPAR, ''))