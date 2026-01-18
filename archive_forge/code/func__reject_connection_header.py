import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _reject_connection_header(headers, hdr_validation_flags):
    """
    Raises a ProtocolError if the Connection header is present in a header
    block.
    """
    for header in headers:
        if header[0] in CONNECTION_HEADERS:
            raise ProtocolError('Connection-specific header field present: %s.' % header[0])
        yield header