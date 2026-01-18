from __future__ import print_function, absolute_import
import io
from .common import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden, PY3
class DefusedGzipDecodedResponse(gzip.GzipFile if gzip else object):
    """a file-like object to decode a response encoded with the gzip
    method, as described in RFC 1952.
    """

    def __init__(self, response, limit=None):
        if not gzip:
            raise NotImplementedError
        self.limit = limit = limit if limit is not None else MAX_DATA
        if limit < 0:
            data = response.read()
            self.readlength = None
        else:
            data = response.read(limit + 1)
            self.readlength = 0
        if limit >= 0 and len(data) > limit:
            raise ValueError('max payload length exceeded')
        self.stringio = io.BytesIO(data)
        gzip.GzipFile.__init__(self, mode='rb', fileobj=self.stringio)

    def read(self, n):
        if self.limit >= 0:
            left = self.limit - self.readlength
            n = min(n, left + 1)
            data = gzip.GzipFile.read(self, n)
            self.readlength += len(data)
            if self.readlength > self.limit:
                raise ValueError('max payload length exceeded')
            return data
        else:
            return gzip.GzipFile.read(self, n)

    def close(self):
        gzip.GzipFile.close(self)
        self.stringio.close()