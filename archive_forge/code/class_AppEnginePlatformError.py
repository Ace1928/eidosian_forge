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
class AppEnginePlatformError(HTTPError):
    pass