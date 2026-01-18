import warnings
from io import StringIO
class IncluderAppIter(Recursive):
    """
    Like Includer, but just stores the app_iter response
    (be sure to call close on the response!)
    """

    def activate(self, environ):
        response = IncludedAppIterResponse()

        def start_response(status, headers, exc_info=None):
            if exc_info:
                raise exc_info
            response.status = status
            response.headers = headers
            return response.write
        app_iter = self.application(environ, start_response)
        response.app_iter = app_iter
        return response