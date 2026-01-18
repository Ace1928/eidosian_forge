from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
from .source import Source
from .token_kind import TokenKind
from ..pyutils import camel_to_snake
class FieldDefinitionNode(DefinitionNode):
    __slots__ = ('description', 'name', 'directives', 'arguments', 'type')
    description: Optional[StringValueNode]
    name: NameNode
    directives: Tuple[ConstDirectiveNode, ...]
    arguments: Tuple['InputValueDefinitionNode', ...]
    type: TypeNode