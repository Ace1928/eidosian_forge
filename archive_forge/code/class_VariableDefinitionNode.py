from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
from .source import Source
from .token_kind import TokenKind
from ..pyutils import camel_to_snake
class VariableDefinitionNode(Node):
    __slots__ = ('variable', 'type', 'default_value', 'directives')
    variable: 'VariableNode'
    type: 'TypeNode'
    default_value: Optional['ConstValueNode']
    directives: Tuple['ConstDirectiveNode', ...]