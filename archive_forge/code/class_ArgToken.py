from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import lexer
import six
class ArgToken(object):
    """Shell token info.

  Attributes:
    value: A string associated with the token.
    token_type: Instance of ArgTokenType
    tree: A subtree of CLI root.
    start: The index of the first char in the original string.
    end: The index directly after the last char in the original string.
  """

    def __init__(self, value, token_type=ArgTokenType.UNKNOWN, tree=None, start=None, end=None):
        self.value = value
        self.token_type = token_type
        self.tree = tree
        self.start = start
        self.end = end

    def __eq__(self, other):
        """Equality based on properties."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        """Improve debugging during tests."""
        return 'ArgToken({}, {}, {}, {})'.format(self.value, self.token_type, self.start, self.end)