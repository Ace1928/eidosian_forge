import copy
import time
import calendar
from ._internal_utils import to_native_string
from .compat import cookielib, urlparse, urlunparse, Morsel, MutableMapping
def _find_no_duplicates(self, name, domain=None, path=None):
    """Both ``__get_item__`` and ``get`` call this function: it's never
        used elsewhere in Requests.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :raises KeyError: if cookie is not found
        :raises CookieConflictError: if there are multiple cookies
            that match name and optionally domain and path
        :return: cookie.value
        """
    toReturn = None
    for cookie in iter(self):
        if cookie.name == name:
            if domain is None or cookie.domain == domain:
                if path is None or cookie.path == path:
                    if toReturn is not None:
                        raise CookieConflictError('There are multiple cookies with name, %r' % name)
                    toReturn = cookie.value
    if toReturn:
        return toReturn
    raise KeyError('name=%r, domain=%r, path=%r' % (name, domain, path))