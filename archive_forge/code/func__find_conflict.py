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
def _find_conflict(context, cached_fields_and_fragment_names, compared_fragments, parent_fields_are_mutually_exclusive, response_name, field1, field2):
    """Determines if there is a conflict between two particular fields."""
    parent_type1, ast1, def1 = field1
    parent_type2, ast2, def2 = field2
    are_mutually_exclusive = parent_fields_are_mutually_exclusive or (parent_type1 != parent_type2 and isinstance(parent_type1, GraphQLObjectType) and isinstance(parent_type2, GraphQLObjectType))
    type1 = def1 and def1.type
    type2 = def2 and def2.type
    if not are_mutually_exclusive:
        name1 = ast1.name.value
        name2 = ast2.name.value
        if name1 != name2:
            return ((response_name, '{} and {} are different fields'.format(name1, name2)), [ast1], [ast2])
        if not _same_arguments(ast1.arguments, ast2.arguments):
            return ((response_name, 'they have differing arguments'), [ast1], [ast2])
    if type1 and type2 and do_types_conflict(type1, type2):
        return ((response_name, 'they return conflicting types {} and {}'.format(type1, type2)), [ast1], [ast2])
    selection_set1 = ast1.selection_set
    selection_set2 = ast2.selection_set
    if selection_set1 and selection_set2:
        conflicts = _find_conflicts_between_sub_selection_sets(context, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, get_named_type(type1), selection_set1, get_named_type(type2), selection_set2)
        return _subfield_conflicts(conflicts, response_name, ast1, ast2)