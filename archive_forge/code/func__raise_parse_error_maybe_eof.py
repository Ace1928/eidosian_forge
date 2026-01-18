import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _raise_parse_error_maybe_eof(self, expected_type, token):
    lex_position = token['start']
    actual_value = token['value']
    actual_type = token['type']
    if actual_type == 'eof':
        raise exceptions.IncompleteExpressionError(lex_position, actual_value, actual_type)
    message = 'Expecting: %s, got: %s' % (expected_type, actual_type)
    raise exceptions.ParseError(lex_position, actual_value, actual_type, message)