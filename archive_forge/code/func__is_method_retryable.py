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
def _is_method_retryable(self, method):
    """Checks if a given HTTP method should be retried upon, depending if
        it is included in the allowed_methods
        """
    if 'method_whitelist' in self.__dict__:
        warnings.warn("Using 'method_whitelist' with Retry is deprecated and will be removed in v2.0. Use 'allowed_methods' instead", DeprecationWarning)
        allowed_methods = self.method_whitelist
    else:
        allowed_methods = self.allowed_methods
    if allowed_methods and method.upper() not in allowed_methods:
        return False
    return True