import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def _truncate_and_ellipsize(value, max_length):
    """
    Truncates the string representation of the specified value to the specified
    maximum length, if necessary. The end of the string is ellipsized if truncation occurs
    """
    value = str(value)
    if len(value) > max_length:
        return value[:max_length - 3] + '...'
    else:
        return value