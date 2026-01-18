import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _raise_http_error(self, url, response, info=None):
    if info is None:
        msg = ''
    else:
        msg = ': ' + info
    raise errors.InvalidHttpResponse(url, 'Unable to handle http code %d%s' % (response.status, msg))