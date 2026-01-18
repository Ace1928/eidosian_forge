class PathInfoForward(ForwardRequestExceptionMiddleware):

    def __call__(self, environ, start_response):
        environ['PATH_INFO'] = p
        return self.app(environ, start_response)