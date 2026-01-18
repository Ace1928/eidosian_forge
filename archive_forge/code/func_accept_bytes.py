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
def accept_bytes(self, bytes):
    self._number_needed_bytes = None
    try:
        _StatefulDecoder.accept_bytes(self, bytes)
    except KeyboardInterrupt:
        raise
    except SmartMessageHandlerError as exception:
        if not isinstance(exception.exc_value, errors.UnknownSmartMethod):
            log_exception_quietly()
        self.message_handler.protocol_error(exception.exc_value)
        self.accept_bytes(b'')
    except Exception as exception:
        self.decoding_failed = True
        if isinstance(exception, errors.UnexpectedProtocolVersionMarker):
            pass
        else:
            log_exception_quietly()
        self.message_handler.protocol_error(exception)