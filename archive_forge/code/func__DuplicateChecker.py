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
def _DuplicateChecker(js):
    result = {}
    for name, value in js:
        if name in result:
            raise ParseError('Failed to load JSON: duplicate key {0}.'.format(name))
        result[name] = value
    return result