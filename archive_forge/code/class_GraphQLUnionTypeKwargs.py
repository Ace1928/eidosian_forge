from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
class GraphQLUnionTypeKwargs(GraphQLNamedTypeKwargs, total=False):
    types: Tuple[GraphQLObjectType, ...]
    resolve_type: Optional[GraphQLTypeResolver]