from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
class KwargBreaker(object):
    """
  Callable that returns true if the supplied token is in the list of keywords,
  ignoring case.
  """

    def __init__(self, kwargs):
        self.kwargs = [kwarg.upper() for kwarg in kwargs]

    def __call__(self, token):
        return token.spelling.upper() in self.kwargs

    def __repr__(self):
        return 'KwargBreaker({})'.format(','.join(self.kwargs))