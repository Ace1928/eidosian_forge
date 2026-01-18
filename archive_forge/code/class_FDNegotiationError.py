from binascii import hexlify
from enum import Enum
import os
from typing import Optional
class FDNegotiationError(AuthenticationError):
    """Raised when file descriptor support is requested but not available"""

    def __init__(self, data):
        super().__init__(data, msg='File descriptor support not available')