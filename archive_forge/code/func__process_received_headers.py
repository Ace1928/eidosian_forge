from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def _process_received_headers(self, headers, header_validation_flags, header_encoding):
    """
        When headers have been received from the remote peer, run a processing
        pipeline on them to transform them into the appropriate form for
        attaching to an event.
        """
    if self.config.normalize_inbound_headers:
        headers = normalize_inbound_headers(headers, header_validation_flags)
    if self.config.validate_inbound_headers:
        headers = validate_headers(headers, header_validation_flags)
    if header_encoding:
        headers = _decode_headers(headers, header_encoding)
    return list(headers)