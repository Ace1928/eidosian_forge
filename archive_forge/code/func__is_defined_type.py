from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _is_defined_type(typename):
    return not _is_introspection_type(typename) and (not _is_builtin_scalar(typename))