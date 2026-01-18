class URLNormalizingMiddleware(object):
    """Middleware filter to handle URL normalization."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        """Normalize URLs."""
        if len(environ['PATH_INFO']) > 1 and environ['PATH_INFO'][-1] == '/':
            environ['PATH_INFO'] = environ['PATH_INFO'].rstrip('/')
        if not environ['PATH_INFO']:
            environ['PATH_INFO'] = '/'
        return self.app(environ, start_response)