import sys
from . import server
from .workers import threadpool
from ._compat import ntob, bton
class Gateway_u0(Gateway_10):
    """A Gateway class to interface HTTPServer with WSGI u.0.

    WSGI u.0 is an experimental protocol, which uses Unicode for keys
    and values in both Python 2 and Python 3.
    """
    version = ('u', 0)

    def get_environ(self):
        """Return a new environ dict targeting the given wsgi.version."""
        req = self.req
        env_10 = super(Gateway_u0, self).get_environ()
        env = dict(env_10.items())
        enc = env.setdefault('wsgi.url_encoding', 'utf-8')
        try:
            env['PATH_INFO'] = req.path.decode(enc)
            env['QUERY_STRING'] = req.qs.decode(enc)
        except UnicodeDecodeError:
            env['wsgi.url_encoding'] = 'ISO-8859-1'
            env['PATH_INFO'] = env_10['PATH_INFO']
            env['QUERY_STRING'] = env_10['QUERY_STRING']
        env.update(env.items())
        return env