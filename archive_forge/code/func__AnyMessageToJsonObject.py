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
def _AnyMessageToJsonObject(self, message):
    """Converts Any message according to Proto3 JSON Specification."""
    if not message.ListFields():
        return {}
    js = OrderedDict()
    type_url = message.type_url
    js['@type'] = type_url
    sub_message = _CreateMessageFromTypeUrl(type_url, self.descriptor_pool)
    sub_message.ParseFromString(message.value)
    message_descriptor = sub_message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
        js['value'] = self._WrapperMessageToJsonObject(sub_message)
        return js
    if full_name in _WKTJSONMETHODS:
        js['value'] = methodcaller(_WKTJSONMETHODS[full_name][0], sub_message)(self)
        return js
    return self._RegularMessageToJsonObject(sub_message, js)