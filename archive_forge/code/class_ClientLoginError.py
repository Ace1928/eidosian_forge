import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
class ClientLoginError(HTTPError):
    """Raised to indicate there was an error authenticating with ClientLogin."""

    def __init__(self, url, code, msg, headers, args):
        HTTPError.__init__(self, url, code, msg, headers, None)
        self.args = args
        self._reason = args.get('Error')
        self.info = args.get('Info')

    def read(self):
        return '%d %s: %s' % (self.code, self.msg, self.reason)

    @property
    def reason(self):
        return self._reason