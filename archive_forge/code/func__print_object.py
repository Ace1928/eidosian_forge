from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_object(type):
    interfaces = type.interfaces
    implemented_interfaces = ' implements {}'.format(', '.join((i.name for i in interfaces))) if interfaces else ''
    return 'type {}{} {{\n{}\n}}'.format(type.name, implemented_interfaces, _print_fields(type))