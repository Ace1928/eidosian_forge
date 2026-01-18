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
def _ConvertListValueMessage(self, value, message, path):
    """Convert a JSON representation into ListValue message."""
    if not isinstance(value, list):
        raise ParseError('ListValue must be in [] which is {0} at {1}'.format(value, path))
    message.ClearField('values')
    for index, item in enumerate(value):
        self._ConvertValueMessage(item, message.values.add(), '{0}[{1}]'.format(path, index))