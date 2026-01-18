import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _validate_host_authority_header(headers):
    """
    Given the :authority and Host headers from a request block that isn't
    a trailer, check that:
     1. At least one of these headers is set.
     2. If both headers are set, they match.

    :param headers: The HTTP header set.
    :raises: ``ProtocolError``
    """
    authority_header_val = None
    host_header_val = None
    for header in headers:
        if header[0] in (b':authority', u':authority'):
            authority_header_val = header[1]
        elif header[0] in (b'host', u'host'):
            host_header_val = header[1]
        yield header
    authority_present = authority_header_val is not None
    host_present = host_header_val is not None
    if not authority_present and (not host_present):
        raise ProtocolError('Request header block does not have an :authority or Host header.')
    if authority_present and host_present:
        if authority_header_val != host_header_val:
            raise ProtocolError('Request header block has mismatched :authority and Host headers: %r / %r' % (authority_header_val, host_header_val))