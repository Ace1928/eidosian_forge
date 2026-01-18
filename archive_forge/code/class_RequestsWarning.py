from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class RequestsWarning(Warning):
    """Base warning for Requests."""