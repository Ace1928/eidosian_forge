import warnings
import sys
from urllib import parse as urlparse
from paste.recursive import ForwardRequestException, RecursiveMiddleware, RecursionLoop
from paste.util import converters
from paste.response import replace_header
class StatusKeeper(object):

    def __init__(self, app, status, url, headers):
        self.app = app
        self.status = status
        self.url = url
        self.headers = headers

    def __call__(self, environ, start_response):

        def keep_status_start_response(status, headers, exc_info=None):
            for header, value in headers:
                if header.lower() == 'set-cookie':
                    self.headers.append((header, value))
                else:
                    replace_header(self.headers, header, value)
            return start_response(self.status, self.headers, exc_info)
        parts = self.url.split('?')
        environ['PATH_INFO'] = parts[0]
        if len(parts) > 1:
            environ['QUERY_STRING'] = parts[1]
        else:
            environ['QUERY_STRING'] = ''
        try:
            return self.app(environ, keep_status_start_response)
        except RecursionLoop as e:
            line = 'Recursion error getting error page: %s\n' % e
            environ['wsgi.errors'].write(line)
            keep_status_start_response('500 Server Error', [('Content-type', 'text/plain')], sys.exc_info())
            body = 'Error: %s.  (Error page could not be fetched)' % self.status
            body = body.encode('utf8')
            return [body]