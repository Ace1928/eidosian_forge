import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers
def _ConvertInteger(value):
    """Convert an integer.

  Args:
    value: A scalar value to convert.

  Returns:
    The integer value.

  Raises:
    ParseError: If an integer couldn't be consumed.
  """
    if isinstance(value, float) and (not value.is_integer()):
        raise ParseError("Couldn't parse integer: {0}".format(value))
    if isinstance(value, str) and value.find(' ') != -1:
        raise ParseError('Couldn\'t parse integer: "{0}"'.format(value))
    if isinstance(value, bool):
        raise ParseError('Bool value {0} is not acceptable for integer field'.format(value))
    return int(value)