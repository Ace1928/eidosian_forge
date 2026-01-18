import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def _become_message(self, message):
    """Assume the non-format-specific state of message."""
    type_specific = getattr(message, '_type_specific_attributes', [])
    for name in message.__dict__:
        if name not in type_specific:
            self.__dict__[name] = message.__dict__[name]