from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
from .source import Source
from .token_kind import TokenKind
from ..pyutils import camel_to_snake
class SchemaDefinitionNode(TypeSystemDefinitionNode):
    __slots__ = ('description', 'directives', 'operation_types')
    description: Optional[StringValueNode]
    directives: Tuple[ConstDirectiveNode, ...]
    operation_types: Tuple['OperationTypeDefinitionNode', ...]