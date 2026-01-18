class EmptyPathRedirectMiddleware:
    """WSGI middleware to redirect from "" to "/"."""

    def __init__(self, application):
        """Initializes this middleware.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
        """
        self._application = application

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        if path:
            return self._application(environ, start_response)
        location = environ.get('SCRIPT_NAME', '') + '/'
        start_response('301 Moved Permanently', [('Location', location)])
        return []