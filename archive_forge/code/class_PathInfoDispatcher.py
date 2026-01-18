import sys
from . import server
from .workers import threadpool
from ._compat import ntob, bton
class PathInfoDispatcher:
    """A WSGI dispatcher for dispatch based on the PATH_INFO."""

    def __init__(self, apps):
        """Initialize path info WSGI app dispatcher.

        Args:
            apps (dict[str,object]|list[tuple[str,object]]): URI prefix
                and WSGI app pairs
        """
        try:
            apps = list(apps.items())
        except AttributeError:
            pass

        def by_path_len(app):
            return len(app[0])
        apps.sort(key=by_path_len, reverse=True)
        self.apps = [(p.rstrip('/'), a) for p, a in apps]

    def __call__(self, environ, start_response):
        """Process incoming WSGI request.

        Ref: :pep:`3333`

        Args:
            environ (Mapping): a dict containing WSGI environment variables
            start_response (callable): function, which sets response
                status and headers

        Returns:
            list[bytes]: iterable containing bytes to be returned in
                HTTP response body

        """
        path = environ['PATH_INFO'] or '/'
        for p, app in self.apps:
            if path.startswith('{path!s}/'.format(path=p)) or path == p:
                environ = environ.copy()
                environ['SCRIPT_NAME'] = environ.get('SCRIPT_NAME', '') + p
                environ['PATH_INFO'] = path[len(p):]
                return app(environ, start_response)
        start_response('404 Not Found', [('Content-Type', 'text/plain'), ('Content-Length', '0')])
        return ['']