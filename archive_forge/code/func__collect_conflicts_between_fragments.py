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
def _collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, fragment_name1, fragment_name2):
    fragment1 = context.get_fragment(fragment_name1)
    fragment2 = context.get_fragment(fragment_name2)
    if not fragment1 or not fragment2:
        return None
    if fragment1 == fragment2:
        return None
    if compared_fragments.has(fragment_name1, fragment_name2, are_mutually_exclusive):
        return None
    compared_fragments.add(fragment_name1, fragment_name2, are_mutually_exclusive)
    field_map1, fragment_names1 = _get_referenced_fields_and_fragment_names(context, cached_fields_and_fragment_names, fragment1)
    field_map2, fragment_names2 = _get_referenced_fields_and_fragment_names(context, cached_fields_and_fragment_names, fragment2)
    _collect_conflicts_between(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, field_map1, field_map2)
    for _fragment_name2 in fragment_names2:
        _collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, fragment_name1, _fragment_name2)
    for _fragment_name1 in fragment_names1:
        _collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, _fragment_name1, fragment_name2)