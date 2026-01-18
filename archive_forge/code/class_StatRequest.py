import os
from ... import urlutils
from . import request
class StatRequest(VfsRequest):

    def do(self, relpath):
        if not relpath.endswith(b'/'):
            relpath += b'/'
        relpath = self.translate_client_path(relpath)
        stat = self._backing_transport.stat(relpath)
        return request.SuccessfulSmartServerResponse((b'stat', str(stat.st_size).encode('ascii'), oct(stat.st_mode).encode('ascii')))