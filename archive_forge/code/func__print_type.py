from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_type(type):
    if isinstance(type, GraphQLScalarType):
        return _print_scalar(type)
    elif isinstance(type, GraphQLObjectType):
        return _print_object(type)
    elif isinstance(type, GraphQLInterfaceType):
        return _print_interface(type)
    elif isinstance(type, GraphQLUnionType):
        return _print_union(type)
    elif isinstance(type, GraphQLEnumType):
        return _print_enum(type)
    assert isinstance(type, GraphQLInputObjectType)
    return _print_input_object(type)