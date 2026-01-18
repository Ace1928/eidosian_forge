import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def HA2(self, entity_body=''):
    """Returns the H(A2) string. See :rfc:`2617` section 3.2.2.3."""
    if self.qop is None or self.qop == 'auth':
        a2 = '%s:%s' % (self.http_method, self.uri)
    elif self.qop == 'auth-int':
        a2 = '%s:%s:%s' % (self.http_method, self.uri, H(entity_body))
    else:
        raise ValueError(self.errmsg('Unrecognized value for qop!'))
    return H(a2)