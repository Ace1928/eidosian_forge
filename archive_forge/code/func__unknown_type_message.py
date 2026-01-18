from ...error import GraphQLError
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
def _unknown_type_message(type, suggested_types):
    message = 'Unknown type "{}".'.format(type)
    if suggested_types:
        message += ' Perhaps you meant {}?'.format(quoted_or_list(suggested_types))
    return message