import os
from ... import urlutils
from . import request
class PutNonAtomicRequest(VfsRequest):

    def do(self, relpath, mode, create_parent, dir_mode):
        relpath = self.translate_client_path(relpath)
        self._relpath = relpath
        self._dir_mode = _deserialise_optional_mode(dir_mode)
        self._mode = _deserialise_optional_mode(mode)
        self._create_parent = create_parent == b'T'

    def do_body(self, body_bytes):
        self._backing_transport.put_bytes_non_atomic(self._relpath, body_bytes, mode=self._mode, create_parent_dir=self._create_parent, dir_mode=self._dir_mode)
        return request.SuccessfulSmartServerResponse((b'ok',))