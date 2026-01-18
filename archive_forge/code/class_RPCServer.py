from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
class RPCServer(object):
    """This wraps the underlying RPC server so we can make a nice error message.

  This will go away once we switch to just using our own http object.
  """

    def __init__(self, original_server):
        """Construct a new rpc server.

    Args:
      original_server: The server to wrap.
    """
        self._server = original_server

    def Send(self, *args, **kwargs):
        try:
            response = self._server.Send(*args, **kwargs)
            log.debug('Got response: %s', response)
            return response
        except urllib.error.HTTPError as e:
            if hasattr(e, 'read'):
                body = e.read()
            else:
                body = ''
            exceptions.reraise(RPCError(e, body=body))