import base64
import functools
import inspect
import json
import logging
import os
import warnings
import six
from six.moves import urllib
def _urlsafe_b64encode(raw_bytes):
    raw_bytes = _to_bytes(raw_bytes, encoding='utf-8')
    return base64.urlsafe_b64encode(raw_bytes).rstrip(b'=')