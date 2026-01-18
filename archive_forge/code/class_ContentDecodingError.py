from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class ContentDecodingError(RequestException, BaseHTTPError):
    """Failed to decode response content."""