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
def _ConvertStructMessage(self, value, message, path):
    """Convert a JSON representation into Struct message."""
    if not isinstance(value, dict):
        raise ParseError('Struct must be in a dict which is {0} at {1}'.format(value, path))
    message.Clear()
    for key in value:
        self._ConvertValueMessage(value[key], message.fields[key], '{0}.{1}'.format(path, key))
    return