class RecursiveMiddleware(object):
    """
    A WSGI middleware that allows for recursive and forwarded calls.
    All these calls go to the same 'application', but presumably that
    application acts differently with different URLs.  The forwarded
    URLs must be relative to this container.
    """

    def __init__(self, application, global_conf=None):
        self.application = application

    def __call__(self, environ, start_response):
        my_script_name = environ.get('SCRIPT_NAME', '')
        environ['pecan.recursive.script_name'] = my_script_name
        try:
            return self.application(environ, start_response)
        except ForwardRequestException as e:
            middleware = CheckForRecursionMiddleware(e.factory(self), environ)
            return middleware(environ, start_response)