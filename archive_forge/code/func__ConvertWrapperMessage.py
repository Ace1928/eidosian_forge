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
def _ConvertWrapperMessage(self, value, message, path):
    """Convert a JSON representation into Wrapper message."""
    field = message.DESCRIPTOR.fields_by_name['value']
    setattr(message, 'value', _ConvertScalarFieldValue(value, field, path='{0}.value'.format(path)))