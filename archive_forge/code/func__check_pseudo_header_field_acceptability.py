import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _check_pseudo_header_field_acceptability(pseudo_headers, method, hdr_validation_flags):
    """
    Given the set of pseudo-headers present in a header block and the
    validation flags, confirms that RFC 7540 allows them.
    """
    if hdr_validation_flags.is_trailer and pseudo_headers:
        raise ProtocolError('Received pseudo-header in trailer %s' % pseudo_headers)
    if hdr_validation_flags.is_response_header:
        _assert_header_in_set(u':status', b':status', pseudo_headers)
        invalid_response_headers = pseudo_headers & _REQUEST_ONLY_HEADERS
        if invalid_response_headers:
            raise ProtocolError('Encountered request-only headers %s' % invalid_response_headers)
    elif not hdr_validation_flags.is_response_header and (not hdr_validation_flags.is_trailer):
        _assert_header_in_set(u':path', b':path', pseudo_headers)
        _assert_header_in_set(u':method', b':method', pseudo_headers)
        _assert_header_in_set(u':scheme', b':scheme', pseudo_headers)
        invalid_request_headers = pseudo_headers & _RESPONSE_ONLY_HEADERS
        if invalid_request_headers:
            raise ProtocolError('Encountered response-only headers %s' % invalid_request_headers)
        if method != b'CONNECT':
            invalid_headers = pseudo_headers & _CONNECT_REQUEST_ONLY_HEADERS
            if invalid_headers:
                raise ProtocolError('Encountered connect-request-only headers %s' % invalid_headers)