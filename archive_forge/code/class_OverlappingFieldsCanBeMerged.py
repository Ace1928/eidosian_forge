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
class OverlappingFieldsCanBeMerged(ValidationRule):
    __slots__ = ('_compared_fragments', '_cached_fields_and_fragment_names')

    def __init__(self, context):
        super(OverlappingFieldsCanBeMerged, self).__init__(context)
        self._compared_fragments = PairSet()
        self._cached_fields_and_fragment_names = {}

    def leave_SelectionSet(self, node, key, parent, path, ancestors):
        conflicts = _find_conflicts_within_selection_set(self.context, self._cached_fields_and_fragment_names, self._compared_fragments, self.context.get_parent_type(), node)
        for (reason_name, reason), fields1, fields2 in conflicts:
            self.context.report_error(GraphQLError(self.fields_conflict_message(reason_name, reason), list(fields1) + list(fields2)))

    @staticmethod
    def same_type(type1, type2):
        return is_equal_type(type1, type2)

    @classmethod
    def fields_conflict_message(cls, reason_name, reason):
        return 'Fields "{}" conflict because {}. Use different aliases on the fields to fetch both if this was intentional.'.format(reason_name, cls.reason_message(reason))

    @classmethod
    def reason_message(cls, reason):
        if isinstance(reason, list):
            return ' and '.join(('subfields "{}" conflict because {}'.format(reason_name, cls.reason_message(sub_reason)) for reason_name, sub_reason in reason))
        return reason