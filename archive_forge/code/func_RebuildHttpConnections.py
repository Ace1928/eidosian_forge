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
def RebuildHttpConnections(http):
    """Rebuilds all http connections in the httplib2.Http instance.

    httplib2 overloads the map in http.connections to contain two different
    types of values:
    { scheme string:  connection class } and
    { scheme + authority string : actual http connection }
    Here we remove all of the entries for actual connections so that on the
    next request httplib2 will rebuild them from the connection types.

    Args:
      http: An httplib2.Http instance.
    """
    if getattr(http, 'connections', None):
        for conn_key in list(http.connections.keys()):
            if ':' in conn_key:
                del http.connections[conn_key]