import errno
import sys
import re
from webob.compat import url_quote
import socket
from webob import exc
from webob.compat import PY2
def _timeout_supported(self, ConnClass):
    if sys.version_info < (2, 7) and ConnClass in (httplib.HTTPConnection, httplib.HTTPSConnection):
        return False
    return True