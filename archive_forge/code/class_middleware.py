import gzip
import io
from paste.response import header_value, remove_header
from paste.httpheaders import CONTENT_LENGTH
class middleware(object):

    def __init__(self, application, compress_level=6):
        self.application = application
        self.compress_level = int(compress_level)

    def __call__(self, environ, start_response):
        if 'gzip' not in environ.get('HTTP_ACCEPT_ENCODING', '') or environ['REQUEST_METHOD'] == 'HEAD':
            return self.application(environ, start_response)
        response = GzipResponse(start_response, self.compress_level)
        app_iter = self.application(environ, response.gzip_start_response)
        if app_iter is not None:
            response.finish_response(app_iter)
        return response.write()