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
def _GetOpener(self):
    """Returns an OpenerDirector that supports cookies and ignores redirects.

    Returns:
      A urllib2.OpenerDirector object.
    """
    opener = OpenerDirector()
    opener.add_handler(ProxyHandler())
    opener.add_handler(UnknownHandler())
    opener.add_handler(HTTPHandler())
    opener.add_handler(HTTPDefaultErrorHandler())
    opener.add_handler(HTTPSHandler())
    opener.add_handler(HTTPErrorProcessor())
    opener.add_handler(ContentEncodingHandler())
    if self.save_cookies:
        self.cookie_jar.filename = os.path.expanduser(HttpRpcServer.DEFAULT_COOKIE_FILE_PATH)
        if os.path.exists(self.cookie_jar.filename):
            try:
                self.cookie_jar.load()
                self.authenticated = True
                logger.debug('Loaded authentication cookies from %s', self.cookie_jar.filename)
            except (OSError, IOError, LoadError) as e:
                logger.debug('Could not load authentication cookies; %s: %s', e.__class__.__name__, e)
                self.cookie_jar.filename = None
        else:
            try:
                fd = os.open(self.cookie_jar.filename, os.O_CREAT, 384)
                os.close(fd)
            except (OSError, IOError) as e:
                logger.debug('Could not create authentication cookies file; %s: %s', e.__class__.__name__, e)
                self.cookie_jar.filename = None
    opener.add_handler(HTTPCookieProcessor(self.cookie_jar))
    return opener