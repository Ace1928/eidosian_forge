import os.path
import socket
from urllib3.poolmanager import PoolManager, proxy_from_url
from urllib3.response import HTTPResponse
from urllib3.util import parse_url
from urllib3.util import Timeout as TimeoutSauce
from urllib3.util.retry import Retry
from urllib3.exceptions import ClosedPoolError
from urllib3.exceptions import ConnectTimeoutError
from urllib3.exceptions import HTTPError as _HTTPError
from urllib3.exceptions import InvalidHeader as _InvalidHeader
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import NewConnectionError
from urllib3.exceptions import ProxyError as _ProxyError
from urllib3.exceptions import ProtocolError
from urllib3.exceptions import ReadTimeoutError
from urllib3.exceptions import SSLError as _SSLError
from urllib3.exceptions import ResponseError
from urllib3.exceptions import LocationValueError
from .models import Response
from .compat import urlparse, basestring
from .utils import (DEFAULT_CA_BUNDLE_PATH, extract_zipped_paths,
from .structures import CaseInsensitiveDict
from .cookies import extract_cookies_to_jar
from .exceptions import (ConnectionError, ConnectTimeout, ReadTimeout, SSLError,
from .auth import _basic_auth_str
def SOCKSProxyManager(*args, **kwargs):
    raise InvalidSchema('Missing dependencies for SOCKS support.')