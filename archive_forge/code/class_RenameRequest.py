import os
from ... import urlutils
from . import request
class RenameRequest(VfsRequest):

    def do(self, rel_from, rel_to):
        rel_from = self.translate_client_path(rel_from)
        rel_to = self.translate_client_path(rel_to)
        self._backing_transport.rename(rel_from, rel_to)
        return request.SuccessfulSmartServerResponse((b'ok',))