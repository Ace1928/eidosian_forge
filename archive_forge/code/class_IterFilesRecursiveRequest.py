import os
from ... import urlutils
from . import request
class IterFilesRecursiveRequest(VfsRequest):

    def do(self, relpath):
        if not relpath.endswith(b'/'):
            relpath += b'/'
        relpath = self.translate_client_path(relpath)
        transport = self._backing_transport.clone(relpath)
        filenames = transport.iter_files_recursive()
        return request.SuccessfulSmartServerResponse((b'names',) + tuple(filenames))