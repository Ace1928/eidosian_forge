import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _ToCamelCase(name):
    """Converts name to camel-case and returns it."""
    capitalize_next = False
    result = []
    for c in name:
        if c == '_':
            if result:
                capitalize_next = True
        elif capitalize_next:
            result.append(c.upper())
            capitalize_next = False
        else:
            result += c
    if result and result[0].isupper():
        result[0] = result[0].lower()
    return ''.join(result)