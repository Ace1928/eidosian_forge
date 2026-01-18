import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def _get_fully_qualified_class_name(obj):
    """
    Obtains the fully qualified class name of the given object.
    """
    return obj.__class__.__module__ + '.' + obj.__class__.__name__