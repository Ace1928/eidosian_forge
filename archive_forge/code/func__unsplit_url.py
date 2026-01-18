import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
@staticmethod
def _unsplit_url(scheme, user, password, host, port, path):
    """Build the full URL for the given already URL encoded path.

        user, password, host and path will be quoted if they contain reserved
        chars.

        Args:
          scheme: protocol
          user: login
          password: associated password
          host: the server address
          port: the associated port
          path: the absolute path on the server

        :return: The corresponding URL.
        """
    netloc = urlutils.quote(host)
    if user is not None:
        netloc = '{}@{}'.format(urlutils.quote(user), netloc)
    if port is not None:
        netloc = '%s:%d' % (netloc, port)
    path = urlutils.escape(path)
    return urlutils.urlparse.urlunparse((scheme, netloc, path, None, None, None))