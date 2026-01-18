import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _check_path_header(headers, hdr_validation_flags):
    """
    Raise a ProtocolError if a header block arrives or is sent that contains an
    empty :path header.
    """

    def inner():
        for header in headers:
            if header[0] in (b':path', u':path'):
                if not header[1]:
                    raise ProtocolError('An empty :path header is forbidden')
            yield header
    skip_validation = hdr_validation_flags.is_response_header or hdr_validation_flags.is_trailer
    if skip_validation:
        return headers
    else:
        return inner()