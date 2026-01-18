import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _ToJsonName(name):
    """Converts name to Json name and returns it."""
    capitalize_next = False
    result = []
    for c in name:
        if c == '_':
            capitalize_next = True
        elif capitalize_next:
            result.append(c.upper())
            capitalize_next = False
        else:
            result += c
    return ''.join(result)