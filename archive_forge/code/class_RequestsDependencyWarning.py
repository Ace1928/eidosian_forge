from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class RequestsDependencyWarning(RequestsWarning):
    """An imported dependency doesn't match the expected version range."""