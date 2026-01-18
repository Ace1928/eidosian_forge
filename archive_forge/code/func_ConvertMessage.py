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
def ConvertMessage(self, value, message, path):
    """Convert a JSON object into a message.

    Args:
      value: A JSON object.
      message: A WKT or regular protocol message to record the data.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of convert problems.
    """
    self.recursion_depth += 1
    if self.recursion_depth > self.max_recursion_depth:
        raise ParseError('Message too deep. Max recursion depth is {0}'.format(self.max_recursion_depth))
    message_descriptor = message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if not path:
        path = message_descriptor.name
    if _IsWrapperMessage(message_descriptor):
        self._ConvertWrapperMessage(value, message, path)
    elif full_name in _WKTJSONMETHODS:
        methodcaller(_WKTJSONMETHODS[full_name][1], value, message, path)(self)
    else:
        self._ConvertFieldValuePair(value, message, path)
    self.recursion_depth -= 1