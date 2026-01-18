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
def _GenericMessageToJsonObject(self, message):
    """Converts message according to Proto3 JSON Specification."""
    return message.ToJsonString()