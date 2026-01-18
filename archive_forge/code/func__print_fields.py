from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_fields(type):
    return '\n'.join(('  {}{}: {}{}'.format(f_name, _print_args(f), f.type, _print_deprecated(f)) for f_name, f in type.fields.items()))