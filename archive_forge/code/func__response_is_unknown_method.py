import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def _response_is_unknown_method(self, result_tuple):
    """Raise UnexpectedSmartServerResponse if the response is an 'unknonwn
        method' response to the request.

        :param response: The response from a smart client call_expecting_body
            call.
        :param verb: The verb used in that call.
        :raises: UnexpectedSmartServerResponse
        """
    if result_tuple == (b'error', b"Generic bzr smart protocol error: bad request '" + self._last_verb + b"'") or result_tuple == (b'error', b"Generic bzr smart protocol error: bad request u'%s'" % self._last_verb):
        self._request.finished_reading()
        raise errors.UnknownSmartMethod(self._last_verb)