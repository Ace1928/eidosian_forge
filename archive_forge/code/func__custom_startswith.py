import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _custom_startswith(test_string, bytes_prefix, unicode_prefix):
    """
    Given a string that might be a bytestring or a Unicode string,
    return True if it starts with the appropriate prefix.
    """
    if isinstance(test_string, bytes):
        return test_string.startswith(bytes_prefix)
    else:
        return test_string.startswith(unicode_prefix)