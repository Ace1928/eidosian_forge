import base64
import os
from anyio.to_thread import run_sync
from tornado import web
from jupyter_server.services.contents.filemanager import (
def _save_large_file(self, os_path, content, format):
    """Save content of a generic file."""
    if format not in {'text', 'base64'}:
        raise web.HTTPError(400, "Must specify format of file contents as 'text' or 'base64'")
    try:
        if format == 'text':
            bcontent = content.encode('utf8')
        else:
            b64_bytes = content.encode('ascii')
            bcontent = base64.b64decode(b64_bytes)
    except Exception as e:
        raise web.HTTPError(400, f'Encoding error saving {os_path}: {e}') from e
    with self.perm_to_403(os_path):
        if os.path.islink(os_path):
            os_path = os.path.join(os.path.dirname(os_path), os.readlink(os_path))
        with open(os_path, 'ab') as f:
            f.write(bcontent)