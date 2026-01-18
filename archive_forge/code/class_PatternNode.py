from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
from cmakelang.parse.argument_nodes import (
class PatternNode(StandardArgTree):
    """Patterns are children of a `PATTERN` keyword argument and are common
     enough to warrent their own node."""

    @classmethod
    def parse(cls, ctx, tokens, breakstack):
        """
    ::

      [PATTERN <pattern> | REGEX <regex>]
      [EXCLUDE] [PERMISSIONS <permissions>...]
    """
        return super(PatternNode, cls).parse(ctx, tokens, npargs='+', kwargs={'PERMISSIONS': PositionalParser('+')}, flags=['EXCLUDE'], breakstack=breakstack)