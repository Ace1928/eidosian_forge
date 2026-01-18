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
def _string_to_bytes(self, message):
    try:
        return message.encode('ascii')
    except UnicodeError:
        raise ValueError('String input must be ASCII-only; use bytes or a Message instead')