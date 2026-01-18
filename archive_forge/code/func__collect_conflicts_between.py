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
def _collect_conflicts_between(context, conflicts, cached_fields_and_fragment_names, compared_fragments, parent_fields_are_mutually_exclusive, field_map1, field_map2):
    """Collect all Conflicts between two collections of fields.

       This is similar to, but different from the `collect_conflicts_within` function above. This check assumes that
       `collect_conflicts_within` has already been called on each provided collection of fields.
       This is true because this validator traverses each individual selection set.
    """
    for response_name, fields1 in list(field_map1.items()):
        fields2 = field_map2.get(response_name)
        if fields2:
            for field1 in fields1:
                for field2 in fields2:
                    conflict = _find_conflict(context, cached_fields_and_fragment_names, compared_fragments, parent_fields_are_mutually_exclusive, response_name, field1, field2)
                    if conflict:
                        conflicts.append(conflict)