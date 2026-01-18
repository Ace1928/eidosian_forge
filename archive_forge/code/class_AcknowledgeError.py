from __future__ import absolute_import
from enum import Enum
from google.api_core.exceptions import GoogleAPICallError
from typing import Optional
class AcknowledgeError(GoogleAPICallError):
    """Error during ack/modack/nack operation on exactly-once-enabled subscription."""

    def __init__(self, error_code: AcknowledgeStatus, info: Optional[str]):
        self.error_code = error_code
        self.info = info
        message = None
        if info:
            message = str(self.error_code) + ' : ' + str(self.info)
        else:
            message = str(self.error_code)
        super(AcknowledgeError, self).__init__(message)