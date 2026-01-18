from collections import Counter
from ...error import GraphQLError
from ...pyutils.ordereddict import OrderedDict
from ...type.definition import (GraphQLInterfaceType, GraphQLObjectType,
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
class FieldsOnCorrectType(ValidationRule):
    """Fields on correct type

      A GraphQL document is only valid if all fields selected are defined by the
      parent type, or are an allowed meta field such as __typenamme
    """

    def enter_Field(self, node, key, parent, path, ancestors):
        parent_type = self.context.get_parent_type()
        if not parent_type:
            return
        field_def = self.context.get_field_def()
        if not field_def:
            schema = self.context.get_schema()
            field_name = node.name.value
            suggested_type_names = get_suggested_type_names(schema, parent_type, field_name)
            suggested_field_names = [] if suggested_type_names else get_suggested_field_names(schema, parent_type, field_name)
            self.context.report_error(GraphQLError(_undefined_field_message(field_name, parent_type.name, suggested_type_names, suggested_field_names), [node]))