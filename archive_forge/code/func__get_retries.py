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
def _get_retries(self, retries, redirect):
    if not isinstance(retries, Retry):
        retries = Retry.from_int(retries, redirect=redirect, default=self.retries)
    if retries.connect or retries.read or retries.redirect:
        warnings.warn('URLFetch only supports total retries and does not recognize connect, read, or redirect retry parameters.', AppEnginePlatformWarning)
    return retries