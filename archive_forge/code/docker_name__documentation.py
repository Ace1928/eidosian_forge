from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import six.moves.urllib.parse
Parses the given name string.

  Parsing is done strictly; registry is required and a Tag will only be returned
  if specified explicitly in the given name string.
  Args:
    name: The name to convert.
  Returns:
    The parsed name.
  Raises:
    BadNameException: The name could not be parsed.
  