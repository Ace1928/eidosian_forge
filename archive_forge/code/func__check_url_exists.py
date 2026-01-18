import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def _check_url_exists(self, url, state):
    global http_client, urlparse, socket
    if http_client is None:
        from http import client as http_client
    if urlparse is None:
        import parse as urlparse
    if socket is None:
        import socket
    scheme, netloc, path, params, query, fragment = urlparse.urlparse(url, 'http')
    if params:
        path += ';' + params
    if query:
        path += '?' + query
    try:
        conn = (http_client.HTTPSConnection if scheme == 'https' else http_client.HTTPConnection)(netloc)
        try:
            conn.request('HEAD', path)
            res = conn.getresponse()
        finally:
            conn.close()
    except http_client.HTTPException as e:
        e = str(e)
        raise Invalid(self.message('httpError', state, error=str(e)), state, url)
    except socket.error as e:
        raise Invalid(self.message('socketError', state, error=str(e)), state, url)
    else:
        if res.status == 404:
            raise Invalid(self.message('notFound', state), state, url)
        if not 200 <= res.status < 500:
            raise Invalid(self.message('status', state, status=res.status), state, url)