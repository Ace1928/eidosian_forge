from __future__ import absolute_import
import io
import logging
import warnings
from ..exceptions import (
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..response import HTTPResponse
from ..util.retry import Retry
from ..util.timeout import Timeout
from . import _appengine_environ
def _get_absolute_timeout(self, timeout):
    if timeout is Timeout.DEFAULT_TIMEOUT:
        return None
    if isinstance(timeout, Timeout):
        if timeout._read is not None or timeout._connect is not None:
            warnings.warn('URLFetch does not support granular timeout settings, reverting to total or default URLFetch timeout.', AppEnginePlatformWarning)
        return timeout.total
    return timeout