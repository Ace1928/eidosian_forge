class DebugMiddleware(object):

    def __init__(self, app, *args, **kwargs):
        self.app = app

    def __call__(self, environ, start_response):
        try:
            return self.app(environ, start_response)
        except Exception as exc:
            out = StringIO()
            print_exc(file=out)
            LOG.exception(exc)
            formatted_environ = pformat(environ)
            result = debug_template.render(traceback=out.getvalue(), environment=formatted_environ)
            response = Response()
            if isinstance(exc, HTTPException):
                response.status_int = exc.status
            else:
                response.status_int = 500
            response.unicode_body = result
            return response(environ, start_response)