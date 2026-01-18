import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def _write_reflog(self, ref, old_sha, new_sha, committer, timestamp, timezone, message):
    from .reflog import format_reflog_line
    path = os.path.join(self.controldir(), 'logs', os.fsdecode(ref))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    if committer is None:
        config = self.get_config_stack()
        committer = self._get_user_identity(config)
    check_user_identity(committer)
    if timestamp is None:
        timestamp = int(time.time())
    if timezone is None:
        timezone = 0
    with open(path, 'ab') as f:
        f.write(format_reflog_line(old_sha, new_sha, committer, timestamp, timezone, message) + b'\n')