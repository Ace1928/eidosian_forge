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
def _find_conflicts_between_sub_selection_sets(context, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, parent_type1, selection_set1, parent_type2, selection_set2):
    """Find all conflicts found between two selection sets.

       Includes those found via spreading in fragments. Called when determining if conflicts exist
       between the sub-fields of two overlapping fields.
    """
    conflicts = []
    field_map1, fragment_names1 = _get_fields_and_fragments_names(context, cached_fields_and_fragment_names, parent_type1, selection_set1)
    field_map2, fragment_names2 = _get_fields_and_fragments_names(context, cached_fields_and_fragment_names, parent_type2, selection_set2)
    _collect_conflicts_between(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, field_map1, field_map2)
    for fragment_name2 in fragment_names2:
        _collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, field_map1, fragment_name2)
    for fragment_name1 in fragment_names1:
        _collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, field_map2, fragment_name1)
    for fragment_name1 in fragment_names1:
        for fragment_name2 in fragment_names2:
            _collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, fragment_name1, fragment_name2)
    return conflicts