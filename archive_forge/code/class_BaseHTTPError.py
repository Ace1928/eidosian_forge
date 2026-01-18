import time
from email.utils import mktime_tz, parsedate_tz
class BaseHTTPError(Exception):
    """
    The base exception class for all HTTP related exceptions.
    """

    def __init__(self, code, message, headers=None):
        self.code = code
        self.message = message
        self.headers = headers
        super().__init__(message)

    def __str__(self):
        return self.message