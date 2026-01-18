from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_input_object(type):
    return 'input {} {{\n{}\n}}'.format(type.name, '\n'.join(('  ' + _print_input_value(name, field) for name, field in type.fields.items())))