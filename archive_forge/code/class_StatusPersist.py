import sys
from .recursive import ForwardRequestException, RecursionLoop
class StatusPersist(object):

    def __init__(self, app, status, url):
        self.app = app
        self.status = status
        self.url = url

    def __call__(self, environ, start_response):

        def keep_status_start_response(status, headers, exc_info=None):
            return start_response(self.status, headers, exc_info)
        parts = self.url.split('?')
        environ['PATH_INFO'] = parts[0]
        if len(parts) > 1:
            environ['QUERY_STRING'] = parts[1]
        else:
            environ['QUERY_STRING'] = ''
        try:
            return self.app(environ, keep_status_start_response)
        except RecursionLoop as e:
            environ['wsgi.errors'].write('Recursion error getting error page: %s\n' % e)
            keep_status_start_response('500 Server Error', [('Content-type', 'text/plain')], sys.exc_info())
            return [b'Error: %s.  (Error page could not be fetched)' % self.status.encode('utf-8')]