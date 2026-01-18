import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _reject_te(headers, hdr_validation_flags):
    """
    Raises a ProtocolError if the TE header is present in a header block and
    its value is anything other than "trailers".
    """
    for header in headers:
        if header[0] in (b'te', u'te'):
            if header[1].lower() not in (b'trailers', u'trailers'):
                raise ProtocolError('Invalid value for Transfer-Encoding header: %s' % header[1])
        yield header