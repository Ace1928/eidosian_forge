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
def _WrapperMessageToJsonObject(self, message):
    return self._FieldToJsonObject(message.DESCRIPTOR.fields_by_name['value'], message.value)