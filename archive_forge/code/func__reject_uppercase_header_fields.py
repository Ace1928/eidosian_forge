import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _reject_uppercase_header_fields(headers, hdr_validation_flags):
    """
    Raises a ProtocolError if any uppercase character is found in a header
    block.
    """
    for header in headers:
        if UPPER_RE.search(header[0]):
            raise ProtocolError('Received uppercase header name %s.' % header[0])
        yield header