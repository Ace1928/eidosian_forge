import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe

        Safely decode an encoded text stream back into a list of messages.

        If the encoded text stream contained an invalid hash or was in an
        invalid format, return None.
        