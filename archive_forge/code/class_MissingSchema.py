from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class MissingSchema(RequestException, ValueError):
    """The URL scheme (e.g. http or https) is missing."""