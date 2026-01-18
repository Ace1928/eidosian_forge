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
def _DevAppServerAuthenticate(self):
    """Authenticates the user on the dev_appserver."""
    credentials = self.auth_function()
    value = self._CreateDevAppServerCookieData(credentials[0], True)
    self.extra_headers['Cookie'] = 'dev_appserver_login="%s"; Path=/;' % value