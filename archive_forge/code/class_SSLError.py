from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class SSLError(ConnectionError):
    """An SSL error occurred."""