import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def _generate_dir_responses(self, path, depth):
    local_path = self.translate_path(path)
    entries = os.listdir(local_path)
    for entry in entries:
        entry_path = urlutils.escape(entry)
        if path.endswith('/'):
            entry_path = path + entry_path
        else:
            entry_path = path + '/' + entry_path
        response, st = self._generate_response(entry_path)
        yield response
        if depth == 'Infinity' and stat.S_ISDIR(st.st_mode):
            yield from self._generate_dir_responses(entry_path, depth)