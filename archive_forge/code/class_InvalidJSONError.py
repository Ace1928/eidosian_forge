from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class InvalidJSONError(RequestException):
    """A JSON error occurred."""