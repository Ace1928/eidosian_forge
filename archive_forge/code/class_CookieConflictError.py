import copy
import time
import calendar
from ._internal_utils import to_native_string
from .compat import cookielib, urlparse, urlunparse, Morsel, MutableMapping
class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific.
    """