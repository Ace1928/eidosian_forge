from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_schema_definition(schema):
    operation_types = []
    query_type = schema.get_query_type()
    if query_type:
        operation_types.append('  query: {}'.format(query_type))
    mutation_type = schema.get_mutation_type()
    if mutation_type:
        operation_types.append('  mutation: {}'.format(mutation_type))
    subscription_type = schema.get_subscription_type()
    if subscription_type:
        operation_types.append('  subscription: {}'.format(subscription_type))
    return 'schema {{\n{}\n}}'.format('\n'.join(operation_types))