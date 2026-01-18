from ...error import GraphQLError
from ...language import ast
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
def _unknown_directive_arg_message(arg_name, directive_name, suggested_args):
    message = 'Unknown argument "{}" on directive "@{}".'.format(arg_name, directive_name)
    if suggested_args:
        message += ' Did you mean {}?'.format(quoted_or_list(suggested_args))
    return message