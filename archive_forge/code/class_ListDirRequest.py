import os
from ... import urlutils
from . import request
class ListDirRequest(VfsRequest):

    def do(self, relpath):
        if not relpath.endswith(b'/'):
            relpath += b'/'
        relpath = self.translate_client_path(relpath)
        filenames = self._backing_transport.list_dir(relpath)
        return request.SuccessfulSmartServerResponse((b'names',) + tuple([filename.encode('utf-8') for filename in filenames]))