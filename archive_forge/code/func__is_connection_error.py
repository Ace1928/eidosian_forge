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
def _is_connection_error(self, err):
    """Errors when we're fairly sure that the server did not receive the
        request, so it should be safe to retry.
        """
    if isinstance(err, ProxyError):
        err = err.original_error
    return isinstance(err, ConnectTimeoutError)