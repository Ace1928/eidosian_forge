import os
from ... import urlutils
from . import request
class MoveRequest(VfsRequest):

    def do(self, rel_from, rel_to):
        rel_from = self.translate_client_path(rel_from)
        rel_to = self.translate_client_path(rel_to)
        self._backing_transport.move(rel_from, rel_to)
        return request.SuccessfulSmartServerResponse((b'ok',))