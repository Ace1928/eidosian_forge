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
def _MessageToJsonObject(self, message):
    """Converts message to an object according to Proto3 JSON Specification."""
    message_descriptor = message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
        return self._WrapperMessageToJsonObject(message)
    if full_name in _WKTJSONMETHODS:
        return methodcaller(_WKTJSONMETHODS[full_name][0], message)(self)
    js = {}
    return self._RegularMessageToJsonObject(message, js)