import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
def _readv(self, relpath, offsets):
    """Get parts of the file at the given relative path.

        :param offsets: A list of (offset, size) tuples.
        :param return: A list or generator of (offset, data) tuples
        """
    offsets = list(offsets)
    try_again = True
    retried_offset = None
    while try_again:
        try_again = False
        sorted_offsets = sorted(offsets)
        coalesced = self._coalesce_offsets(sorted_offsets, limit=self._max_readv_combine, fudge_factor=self._bytes_to_read_before_seek, max_size=self._get_max_size)
        coalesced = list(coalesced)
        if 'http' in debug.debug_flags:
            mutter('http readv of %s  offsets => %s collapsed %s', relpath, len(offsets), len(coalesced))
        data_map = {}
        iter_offsets = iter(offsets)
        try:
            cur_offset_and_size = next(iter_offsets)
        except StopIteration:
            return
        try:
            for cur_coal, rfile in self._coalesce_readv(relpath, coalesced):
                for offset, size in cur_coal.ranges:
                    start = cur_coal.start + offset
                    rfile.seek(start, os.SEEK_SET)
                    data = rfile.read(size)
                    data_len = len(data)
                    if data_len != size:
                        raise errors.ShortReadvError(relpath, start, size, actual=data_len)
                    if (start, size) == cur_offset_and_size:
                        yield (cur_offset_and_size[0], data)
                        try:
                            cur_offset_and_size = next(iter_offsets)
                        except StopIteration:
                            return
                    else:
                        data_map[start, size] = data
                while cur_offset_and_size in data_map:
                    this_data = data_map.pop(cur_offset_and_size)
                    yield (cur_offset_and_size[0], this_data)
                    try:
                        cur_offset_and_size = next(iter_offsets)
                    except StopIteration:
                        return
        except (errors.ShortReadvError, errors.InvalidRange, errors.InvalidHttpRange, errors.HttpBoundaryMissing) as e:
            mutter('Exception %r: %s during http._readv', e, e)
            if not isinstance(e, errors.ShortReadvError) or retried_offset == cur_offset_and_size:
                if not self._degrade_range_hint(relpath, coalesced):
                    raise
            offsets = [cur_offset_and_size] + [o for o in iter_offsets]
            retried_offset = cur_offset_and_size
            try_again = True