from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_directive(directive):
    return 'directive @{}{} on {}'.format(directive.name, _print_args(directive), ' | '.join(directive.locations))