import itertools
from collections import OrderedDict
from ...error import GraphQLError
from ...language import ast
from ...language.printer import print_ast
from ...pyutils.pair_set import PairSet
from ...type.definition import (GraphQLInterfaceType, GraphQLList,
from ...utils.type_comparators import is_equal_type
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
def _collect_fields_and_fragment_names(context, parent_type, selection_set, ast_and_defs, fragment_names):
    for selection in selection_set.selections:
        if isinstance(selection, ast.Field):
            field_name = selection.name.value
            if isinstance(parent_type, (GraphQLObjectType, GraphQLInterfaceType)):
                field_def = parent_type.fields.get(field_name)
            else:
                field_def = None
            response_name = selection.alias.value if selection.alias else field_name
            if not ast_and_defs.get(response_name):
                ast_and_defs[response_name] = []
            ast_and_defs[response_name].append([parent_type, selection, field_def])
        elif isinstance(selection, ast.FragmentSpread):
            fragment_names[selection.name.value] = True
        elif isinstance(selection, ast.InlineFragment):
            type_condition = selection.type_condition
            if type_condition:
                inline_fragment_type = type_from_ast(context.get_schema(), selection.type_condition)
            else:
                inline_fragment_type = parent_type
            _collect_fields_and_fragment_names(context, inline_fragment_type, selection.selection_set, ast_and_defs, fragment_names)