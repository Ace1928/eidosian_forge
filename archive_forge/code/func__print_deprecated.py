from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_deprecated(field_or_enum_value):
    reason = field_or_enum_value.deprecation_reason
    if reason is None:
        return ''
    elif reason in ('', DEFAULT_DEPRECATION_REASON):
        return ' @deprecated'
    else:
        return ' @deprecated(reason: {})'.format(print_ast(ast_from_value(reason)))