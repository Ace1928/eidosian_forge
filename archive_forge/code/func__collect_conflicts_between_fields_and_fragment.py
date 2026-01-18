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
def _collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, field_map, fragment_name):
    fragment = context.get_fragment(fragment_name)
    if not fragment:
        return None
    field_map2, fragment_names2 = _get_referenced_fields_and_fragment_names(context, cached_fields_and_fragment_names, fragment)
    _collect_conflicts_between(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, field_map, field_map2)
    for fragment_name2 in fragment_names2:
        _collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragments, are_mutually_exclusive, field_map, fragment_name2)