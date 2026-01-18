import gzip
import io
from paste.response import header_value, remove_header
from paste.httpheaders import CONTENT_LENGTH
class GzipResponse(object):

    def __init__(self, start_response, compress_level):
        self.start_response = start_response
        self.compress_level = compress_level
        self.buffer = io.BytesIO()
        self.compressible = False
        self.content_length = None

    def gzip_start_response(self, status, headers, exc_info=None):
        self.headers = headers
        ct = header_value(headers, 'content-type')
        ce = header_value(headers, 'content-encoding')
        self.compressible = False
        if ct and (ct.startswith('text/') or ct.startswith('application/')) and ('zip' not in ct):
            self.compressible = True
        if ce:
            self.compressible = False
        if self.compressible:
            headers.append(('content-encoding', 'gzip'))
        remove_header(headers, 'content-length')
        self.headers = headers
        self.status = status
        return self.buffer.write

    def write(self):
        out = self.buffer
        out.seek(0)
        s = out.getvalue()
        out.close()
        return [s]

    def finish_response(self, app_iter):
        if self.compressible:
            output = gzip.GzipFile(mode='wb', compresslevel=self.compress_level, fileobj=self.buffer)
        else:
            output = self.buffer
        try:
            for s in app_iter:
                output.write(s)
            if self.compressible:
                output.close()
        finally:
            if hasattr(app_iter, 'close'):
                app_iter.close()
        content_length = self.buffer.tell()
        CONTENT_LENGTH.update(self.headers, content_length)
        self.start_response(self.status, self.headers)