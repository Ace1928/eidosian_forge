import sys
import os
import hotshot
import hotshot.stats
import threading
import cgi
import time
from io import StringIO
from paste import response
class ProfileMiddleware(object):
    """
    Middleware that profiles all requests.

    All HTML pages will have profiling information appended to them.
    The data is isolated to that single request, and does not include
    data from previous requests.

    This uses the ``hotshot`` module, which affects performance of the
    application.  It also runs in a single-threaded mode, so it is
    only usable in development environments.
    """
    style = 'clear: both; background-color: #ff9; color: #000; border: 2px solid #000; padding: 5px;'

    def __init__(self, app, global_conf=None, log_filename='profile.log.tmp', limit=40):
        self.app = app
        self.lock = threading.Lock()
        self.log_filename = log_filename
        self.limit = limit

    def __call__(self, environ, start_response):
        catch_response = []
        body = []

        def replace_start_response(status, headers, exc_info=None):
            catch_response.extend([status, headers])
            start_response(status, headers, exc_info)
            return body.append

        def run_app():
            app_iter = self.app(environ, replace_start_response)
            try:
                body.extend(app_iter)
            finally:
                if hasattr(app_iter, 'close'):
                    app_iter.close()
        self.lock.acquire()
        try:
            prof = hotshot.Profile(self.log_filename)
            prof.addinfo('URL', environ.get('PATH_INFO', ''))
            try:
                prof.runcall(run_app)
            finally:
                prof.close()
            body = ''.join(body)
            headers = catch_response[1]
            content_type = response.header_value(headers, 'content-type')
            if content_type is None or not content_type.startswith('text/html'):
                return [body]
            stats = hotshot.stats.load(self.log_filename)
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            output = capture_output(stats.print_stats, self.limit)
            output_callers = capture_output(stats.print_callers, self.limit)
            body += '<pre style="%s">%s\n%s</pre>' % (self.style, cgi.escape(output), cgi.escape(output_callers))
            return [body]
        finally:
            self.lock.release()