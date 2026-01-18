from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
import uuid
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _handle_existing_file(self, conn, source, dest, proto, timeout):
    """
        Determines whether the source and destination file match.

        :return: False if source and dest both exist and have matching sha1 sums, True otherwise.
        """
    if not os.path.exists(dest):
        return True
    cwd = self._loader.get_basedir()
    filename = str(uuid.uuid4())
    tmp_dest_file = os.path.join(cwd, filename)
    try:
        conn.get_file(source=source, destination=tmp_dest_file, proto=proto, timeout=timeout)
    except ConnectionError as exc:
        error = to_text(exc)
        if error.endswith('No such file or directory'):
            if os.path.exists(tmp_dest_file):
                os.remove(tmp_dest_file)
            return True
    try:
        with open(tmp_dest_file, 'r') as f:
            new_content = f.read()
        with open(dest, 'r') as f:
            old_content = f.read()
    except (IOError, OSError):
        os.remove(tmp_dest_file)
        raise
    sha1 = hashlib.sha1()
    old_content_b = to_bytes(old_content, errors='surrogate_or_strict')
    sha1.update(old_content_b)
    checksum_old = sha1.digest()
    sha1 = hashlib.sha1()
    new_content_b = to_bytes(new_content, errors='surrogate_or_strict')
    sha1.update(new_content_b)
    checksum_new = sha1.digest()
    os.remove(tmp_dest_file)
    if checksum_old == checksum_new:
        return False
    return True