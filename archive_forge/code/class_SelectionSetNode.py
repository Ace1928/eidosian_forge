from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
from .source import Source
from .token_kind import TokenKind
from ..pyutils import camel_to_snake
class SelectionSetNode(Node):
    __slots__ = ('selections',)
    selections: Tuple['SelectionNode', ...]