import collections
import contextlib
import logging
import socket
import time
import httplib2
import six
from six.moves import http_client
from six.moves.urllib import parse
from apitools.base.py import exceptions
from apitools.base.py import util
@contextlib.contextmanager
def _Httplib2Debuglevel(http_request, level, http=None):
    """Temporarily change the value of httplib2.debuglevel, if necessary.

    If http_request has a `loggable_body` distinct from `body`, then we
    need to prevent httplib2 from logging the full body. This sets
    httplib2.debuglevel for the duration of the `with` block; however,
    that alone won't change the value of existing HTTP connections. If
    an httplib2.Http object is provided, we'll also change the level on
    any cached connections attached to it.

    Args:
      http_request: a Request we're logging.
      level: (int) the debuglevel for logging.
      http: (optional) an httplib2.Http whose connections we should
        set the debuglevel on.

    Yields:
      None.
    """
    if http_request.loggable_body is None:
        yield
        return
    old_level = httplib2.debuglevel
    http_levels = {}
    httplib2.debuglevel = level
    if http is not None:
        for connection_key, connection in http.connections.items():
            if ':' not in connection_key:
                continue
            http_levels[connection_key] = connection.debuglevel
            connection.set_debuglevel(level)
    yield
    httplib2.debuglevel = old_level
    if http is not None:
        for connection_key, old_level in http_levels.items():
            if connection_key in http.connections:
                http.connections[connection_key].set_debuglevel(old_level)