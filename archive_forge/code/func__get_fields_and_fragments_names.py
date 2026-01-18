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
def _get_fields_and_fragments_names(context, cached_fields_and_fragment_names, parent_type, selection_set):
    cached = cached_fields_and_fragment_names.get(selection_set)
    if not cached:
        ast_and_defs = OrderedDict()
        fragment_names = OrderedDict()
        _collect_fields_and_fragment_names(context, parent_type, selection_set, ast_and_defs, fragment_names)
        cached = [ast_and_defs, list(fragment_names.keys())]
        cached_fields_and_fragment_names[selection_set] = cached
    return cached