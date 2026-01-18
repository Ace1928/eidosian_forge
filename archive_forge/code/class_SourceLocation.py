from __future__ import print_function
from __future__ import unicode_literals
import re
import sys
from cmakelang import common
class SourceLocation(tuple):
    """
  Named tuple of (line, col, offset)
  """

    @property
    def line(self):
        return self[0]

    @property
    def col(self):
        return self[1]

    @property
    def offset(self):
        return self[2]

    def __repr__(self):
        return '{}:{}'.format(self.line, self.col)