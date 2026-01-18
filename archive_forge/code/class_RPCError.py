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
class RPCError(Error):
    """For when an error occurs when making an RPC call."""

    def __init__(self, url_error, body=''):
        super(RPCError, self).__init__('Server responded with code [{code}]:\n  {reason}.\n  {body}'.format(code=url_error.code, reason=getattr(url_error, 'reason', '(unknown)'), body=body))
        self.url_error = url_error