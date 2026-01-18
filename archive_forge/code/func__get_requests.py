import bisect
import errno
import itertools
import os
import random
import stat
import sys
import time
import warnings
from .. import config, debug, errors, urlutils
from ..errors import LockError, ParamikoNotPresent, PathError, TransportError
from ..osutils import fancy_rename
from ..trace import mutter, warning
from ..transport import (ConnectedTransport, FileExists, FileFileStream,
def _get_requests(self):
    """Break up the offsets into individual requests over sftp.

        The SFTP spec only requires implementers to support 32kB requests. We
        could try something larger (openssh supports 64kB), but then we have to
        handle requests that fail.
        So instead, we just break up our maximum chunks into 32kB chunks, and
        asyncronously requests them.
        Newer versions of paramiko would do the chunking for us, but we want to
        start processing results right away, so we do it ourselves.
        """
    sorted_offsets = sorted(self.original_offsets)
    coalesced = list(ConnectedTransport._coalesce_offsets(sorted_offsets, limit=0, fudge_factor=0))
    requests = []
    for c_offset in coalesced:
        start = c_offset.start
        size = c_offset.length
        while size > 0:
            next_size = min(size, self._max_request_size)
            requests.append((start, next_size))
            size -= next_size
            start += next_size
    if 'sftp' in debug.debug_flags:
        mutter('SFTP.readv(%s) %s offsets => %s coalesced => %s requests', self.relpath, len(sorted_offsets), len(coalesced), len(requests))
    return requests