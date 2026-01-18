from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
class ParenBreaker(object):
    """
  Callable that returns true if the supplied token is a right parenthential
  """

    def __call__(self, token):
        return token.type == lex.TokenType.RIGHT_PAREN