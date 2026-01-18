from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
class CGIHandler(BaseCGIHandler):
    """CGI-based invocation via sys.stdin/stdout/stderr and os.environ

    Usage::

        CGIHandler().run(app)

    The difference between this class and BaseCGIHandler is that it always
    uses 'wsgi.run_once' of 'True', 'wsgi.multithread' of 'False', and
    'wsgi.multiprocess' of 'True'.  It does not take any initialization
    parameters, but always uses 'sys.stdin', 'os.environ', and friends.

    If you need to override any of these parameters, use BaseCGIHandler
    instead.
    """
    wsgi_run_once = True
    os_environ = {}

    def __init__(self):
        BaseCGIHandler.__init__(self, sys.stdin.buffer, sys.stdout.buffer, sys.stderr, read_environ(), multithread=False, multiprocess=True)