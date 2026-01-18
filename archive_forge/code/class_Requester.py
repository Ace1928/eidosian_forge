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
class Requester:
    """Abstract base class for an object that can issue requests on a smart
    medium.
    """

    def call(self, *args):
        """Make a remote call.

        :param args: the arguments of this call.
        """
        raise NotImplementedError(self.call)

    def call_with_body_bytes(self, args, body):
        """Make a remote call with a body.

        :param args: the arguments of this call.
        :type body: str
        :param body: the body to send with the request.
        """
        raise NotImplementedError(self.call_with_body_bytes)

    def call_with_body_readv_array(self, args, body):
        """Make a remote call with a readv array.

        :param args: the arguments of this call.
        :type body: iterable of (start, length) tuples.
        :param body: the readv ranges to send with this request.
        """
        raise NotImplementedError(self.call_with_body_readv_array)

    def set_headers(self, headers):
        raise NotImplementedError(self.set_headers)