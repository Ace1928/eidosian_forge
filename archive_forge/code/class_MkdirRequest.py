import os
from ... import urlutils
from . import request
class MkdirRequest(VfsRequest):

    def do(self, relpath, mode):
        relpath = self.translate_client_path(relpath)
        self._backing_transport.mkdir(relpath, _deserialise_optional_mode(mode))
        return request.SuccessfulSmartServerResponse((b'ok',))