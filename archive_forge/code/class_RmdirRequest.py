import os
from ... import urlutils
from . import request
class RmdirRequest(VfsRequest):

    def do(self, relpath):
        relpath = self.translate_client_path(relpath)
        self._backing_transport.rmdir(relpath)
        return request.SuccessfulSmartServerResponse((b'ok',))