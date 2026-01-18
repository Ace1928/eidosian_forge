import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _OptionsOrNone(descriptor_proto):
    """Returns the value of the field `options`, or None if it is not set."""
    if descriptor_proto.HasField('options'):
        return descriptor_proto.options
    else:
        return None