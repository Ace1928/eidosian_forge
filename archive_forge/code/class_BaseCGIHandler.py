from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
class BaseCGIHandler(SimpleHandler):
    """CGI-like systems using input/output/error streams and environ mapping

    Usage::

        handler = BaseCGIHandler(inp,out,err,env)
        handler.run(app)

    This handler class is useful for gateway protocols like ReadyExec and
    FastCGI, that have usable input/output/error streams and an environment
    mapping.  It's also the base class for CGIHandler, which just uses
    sys.stdin, os.environ, and so on.

    The constructor also takes keyword arguments 'multithread' and
    'multiprocess' (defaulting to 'True' and 'False' respectively) to control
    the configuration sent to the application.  It sets 'origin_server' to
    False (to enable CGI-like output), and assumes that 'wsgi.run_once' is
    False.
    """
    origin_server = False