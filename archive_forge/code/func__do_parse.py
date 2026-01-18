import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _do_parse(self, expression):
    try:
        return self._parse(expression)
    except exceptions.LexerError as e:
        e.expression = expression
        raise
    except exceptions.IncompleteExpressionError as e:
        e.set_expression(expression)
        raise
    except exceptions.ParseError as e:
        e.expression = expression
        raise