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
@DEFAULT_REDIRECT_HEADERS_BLACKLIST.setter
def DEFAULT_REDIRECT_HEADERS_BLACKLIST(cls, value):
    warnings.warn("Using 'Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT' instead", DeprecationWarning)
    cls.DEFAULT_REMOVE_HEADERS_ON_REDIRECT = value