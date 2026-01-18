import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _reject_pseudo_header_fields(headers, hdr_validation_flags):
    """
    Raises a ProtocolError if duplicate pseudo-header fields are found in a
    header block or if a pseudo-header field appears in a block after an
    ordinary header field.

    Raises a ProtocolError if pseudo-header fields are found in trailers.
    """
    seen_pseudo_header_fields = set()
    seen_regular_header = False
    method = None
    for header in headers:
        if _custom_startswith(header[0], b':', u':'):
            if header[0] in seen_pseudo_header_fields:
                raise ProtocolError('Received duplicate pseudo-header field %s' % header[0])
            seen_pseudo_header_fields.add(header[0])
            if seen_regular_header:
                raise ProtocolError('Received pseudo-header field out of sequence: %s' % header[0])
            if header[0] not in _ALLOWED_PSEUDO_HEADER_FIELDS:
                raise ProtocolError('Received custom pseudo-header field %s' % header[0])
            if header[0] in (b':method', u':method'):
                if not isinstance(header[1], bytes):
                    method = header[1].encode('utf-8')
                else:
                    method = header[1]
        else:
            seen_regular_header = True
        yield header
    _check_pseudo_header_field_acceptability(seen_pseudo_header_fields, method, hdr_validation_flags)