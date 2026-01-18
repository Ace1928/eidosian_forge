from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
from ..packages import six
@DEFAULT_METHOD_WHITELIST.setter
def DEFAULT_METHOD_WHITELIST(cls, value):
    warnings.warn("Using 'Retry.DEFAULT_METHOD_WHITELIST' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_ALLOWED_METHODS' instead", DeprecationWarning)
    cls.DEFAULT_ALLOWED_METHODS = value