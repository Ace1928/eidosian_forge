import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _resolve_dot_segments(path):
    """Normalize the URL path by resolving segments of '.' and '..'. For
    more details, see `RFC 3986 section 5.2.4, Remove Dot Segments`_.

    Args:
       path: sequence of path segments in text form

    Returns:
       A new sequence of path segments with the '.' and '..' elements removed
           and resolved.

    .. _RFC 3986 section 5.2.4, Remove Dot Segments: https://tools.ietf.org/html/rfc3986#section-5.2.4
    """
    segs = []
    for seg in path:
        if seg == u'.':
            pass
        elif seg == u'..':
            if segs:
                segs.pop()
        else:
            segs.append(seg)
    if list(path[-1:]) in ([u'.'], [u'..']):
        segs.append(u'')
    return segs