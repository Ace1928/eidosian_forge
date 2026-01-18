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
def _ConvertGenericMessage(self, value, message, path):
    """Convert a JSON representation into message with FromJsonString."""
    try:
        message.FromJsonString(value)
    except ValueError as e:
        raise ParseError('{0} at {1}'.format(e, path)) from e