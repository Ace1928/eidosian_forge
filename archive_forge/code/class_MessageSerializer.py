import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe
class MessageSerializer:

    def loads(self, data):
        return json.loads(data.decode('latin-1'), cls=MessageDecoder)