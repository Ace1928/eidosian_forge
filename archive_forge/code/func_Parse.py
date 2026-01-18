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
def Parse(text, message, ignore_unknown_fields=False, descriptor_pool=None, max_recursion_depth=100):
    """Parses a JSON representation of a protocol message into a message.

  Args:
    text: Message JSON representation.
    message: A protocol buffer message to merge into.
    ignore_unknown_fields: If True, do not raise errors for unknown fields.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    max_recursion_depth: max recursion depth of JSON message to be deserialized.
      JSON messages over this depth will fail to be deserialized. Default value
      is 100.

  Returns:
    The same message passed as argument.

  Raises::
    ParseError: On JSON parsing problems.
  """
    if not isinstance(text, str):
        text = text.decode('utf-8')
    try:
        js = json.loads(text, object_pairs_hook=_DuplicateChecker)
    except ValueError as e:
        raise ParseError('Failed to load JSON: {0}.'.format(str(e))) from e
    return ParseDict(js, message, ignore_unknown_fields, descriptor_pool, max_recursion_depth)