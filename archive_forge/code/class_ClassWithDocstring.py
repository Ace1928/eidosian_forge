from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class ClassWithDocstring(object):
    """Test class for testing help text output.

  This is some detail description of this test class.
  """

    def __init__(self, message='Hello!'):
        """Constructor of the test class.

    Constructs a new ClassWithDocstring object.

    Args:
      message: The default message to print.
    """
        self.message = message

    def print_msg(self, msg=None):
        """Prints a message."""
        if msg is None:
            msg = self.message
        print(msg)