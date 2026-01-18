from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
class IISCGIHandler(BaseCGIHandler):
    """CGI-based invocation with workaround for IIS path bug

    This handler should be used in preference to CGIHandler when deploying on
    Microsoft IIS without having set the config allowPathInfo option (IIS>=7)
    or metabase allowPathInfoForScriptMappings (IIS<7).
    """
    wsgi_run_once = True
    os_environ = {}

    def __init__(self):
        environ = read_environ()
        path = environ.get('PATH_INFO', '')
        script = environ.get('SCRIPT_NAME', '')
        if (path + '/').startswith(script + '/'):
            environ['PATH_INFO'] = path[len(script):]
        BaseCGIHandler.__init__(self, sys.stdin.buffer, sys.stdout.buffer, sys.stderr, environ, multithread=False, multiprocess=True)