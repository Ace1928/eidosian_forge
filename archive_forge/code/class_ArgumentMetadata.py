from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
class ArgumentMetadata(object):
    """Encapsulates metadata about a single argument.

  Attributes:
    arg_name: The name of the argument.
    arg_value: Value of the argument.
    scope: The scope of the argument.
    start_index: The  index where the argument starts in the command string.
    stop_index: The index where the argument stops in the command string.
  """

    def __init__(self, arg_name, arg_value, scope, start_index, stop_index):
        self.arg_name = arg_name
        self.arg_value = arg_value
        self.scope = scope
        self.start_index = start_index
        self.stop_index = stop_index

    def __str__(self):
        """Returns a human-readable representation of an argument's metadata."""
        return 'ArgumentMetadata(name={name},  value={value},  scope={scope},  start_index={start},  stop_index={stop})'.format(name=self.arg_name, scope=self.scope, value=self.arg_value, start=self.start_index, stop=self.stop_index)

    def __eq__(self, other):
        if isinstance(other, ArgumentMetadata):
            return self.arg_name == other.arg_name and self.arg_value == other.arg_value and (self.scope == other.scope) and (self.start_index == other.start_index) and (self.stop_index == other.stop_index)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)