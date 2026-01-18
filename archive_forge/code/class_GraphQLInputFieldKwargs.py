from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
class GraphQLInputFieldKwargs(TypedDict, total=False):
    type_: 'GraphQLInputType'
    default_value: Any
    description: Optional[str]
    deprecation_reason: Optional[str]
    out_name: Optional[str]
    extensions: Dict[str, Any]
    ast_node: Optional[InputValueDefinitionNode]