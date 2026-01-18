from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
class SimpleHandler(BaseHandler):
    """Handler that's just initialized with streams, environment, etc.

    This handler subclass is intended for synchronous HTTP/1.0 origin servers,
    and handles sending the entire response output, given the correct inputs.

    Usage::

        handler = SimpleHandler(
            inp,out,err,env, multithread=False, multiprocess=True
        )
        handler.run(app)"""

    def __init__(self, stdin, stdout, stderr, environ, multithread=True, multiprocess=False):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.base_env = environ
        self.wsgi_multithread = multithread
        self.wsgi_multiprocess = multiprocess

    def get_stdin(self):
        return self.stdin

    def get_stderr(self):
        return self.stderr

    def add_cgi_vars(self):
        self.environ.update(self.base_env)

    def _write(self, data):
        result = self.stdout.write(data)
        if result is None or result == len(data):
            return
        from warnings import warn
        warn('SimpleHandler.stdout.write() should not do partial writes', DeprecationWarning)
        while True:
            data = data[result:]
            if not data:
                break
            result = self.stdout.write(data)

    def _flush(self):
        self.stdout.flush()
        self._flush = self.stdout.flush