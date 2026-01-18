from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_filtered_schema(schema, directive_filter, type_filter):
    return '\n\n'.join([_print_schema_definition(schema)] + [_print_directive(directive) for directive in schema.get_directives() if directive_filter(directive.name)] + [_print_type(type) for typename, type in sorted(schema.get_type_map().items()) if type_filter(typename)]) + '\n'