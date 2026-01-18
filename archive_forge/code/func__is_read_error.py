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
def _is_read_error(self, err):
    """Errors that occur after the request has been started, so we should
        assume that the server began processing it.
        """
    return isinstance(err, (ReadTimeoutError, ProtocolError))