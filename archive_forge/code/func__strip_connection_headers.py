import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _strip_connection_headers(headers, hdr_validation_flags):
    """
    Strip any connection headers as per RFC7540 § 8.1.2.2.
    """
    for header in headers:
        if header[0] not in CONNECTION_HEADERS:
            yield header