from collections import defaultdict
from functools import cmp_to_key
from typing import Any, Dict, List, Union, cast
from ...type import (
from ...error import GraphQLError
from ...language import FieldNode
from ...pyutils import did_you_mean, natural_comparison_key, suggestion_list
from . import ValidationRule
class FieldsOnCorrectTypeRule(ValidationRule):
    """Fields on correct type

    A GraphQL document is only valid if all fields selected are defined by the parent
    type, or are an allowed meta field such as ``__typename``.

    See https://spec.graphql.org/draft/#sec-Field-Selections
    """

    def enter_field(self, node: FieldNode, *_args: Any) -> None:
        type_ = self.context.get_parent_type()
        if not type_:
            return
        field_def = self.context.get_field_def()
        if field_def:
            return
        schema = self.context.schema
        field_name = node.name.value
        suggestion = did_you_mean(get_suggested_type_names(schema, type_, field_name), 'to use an inline fragment on')
        if not suggestion:
            suggestion = did_you_mean(get_suggested_field_names(type_, field_name))
        self.report_error(GraphQLError(f"Cannot query field '{field_name}' on type '{type_}'." + suggestion, node))