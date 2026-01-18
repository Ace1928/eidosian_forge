from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_interface(type):
    return 'interface {} {{\n{}\n}}'.format(type.name, _print_fields(type))