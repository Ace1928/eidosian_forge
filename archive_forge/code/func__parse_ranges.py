import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
def _parse_ranges(self, ranges_header, file_size):
    """Parse the range header value and returns ranges.

        RFC2616 14.35 says that syntactically invalid range specifiers MUST be
        ignored. In that case, we return None instead of a range list.

        :param ranges_header: The 'Range' header value.

        :param file_size: The size of the requested file.

        :return: A list of (start, end) tuples or None if some invalid range
            specifier is encountered.
        """
    if not ranges_header.startswith('bytes='):
        return None
    tail = None
    ranges = []
    ranges_header = ranges_header[len('bytes='):]
    for range_str in ranges_header.split(','):
        range_match = self._range_regexp.match(range_str)
        if range_match is not None:
            start = int(range_match.group('start'))
            end_match = range_match.group('end')
            if end_match is None:
                end = file_size
            else:
                end = int(end_match)
            if start > end:
                return None
            ranges.append((start, end))
        else:
            tail_match = self._tail_regexp.match(range_str)
            if tail_match is not None:
                tail = int(tail_match.group('tail'))
            else:
                return None
    if tail is not None:
        ranges.append((max(0, file_size - tail), file_size))
    checked_ranges = []
    for start, end in ranges:
        if start >= file_size:
            return None
        end = min(end, file_size - 1)
        checked_ranges.append((start, end))
    return checked_ranges