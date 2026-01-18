import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _reuse_for(self, other_base):
    """Returns a transport sharing the same connection if possible.

        Note: we share the connection if the expected credentials are the
        same: (host, port, user). Some protocols may disagree and redefine the
        criteria in daughter classes.

        Note: we don't compare the passwords here because other_base may have
        been obtained from an existing transport.base which do not mention the
        password.

        :param other_base: the URL we want to share the connection with.

        :return: A new transport or None if the connection cannot be shared.
        """
    try:
        parsed_url = self._split_url(other_base)
    except urlutils.InvalidURL:
        return None
    transport = None
    if parsed_url.scheme == self._parsed_url.scheme and parsed_url.user == self._parsed_url.user and (parsed_url.host == self._parsed_url.host) and (parsed_url.port == self._parsed_url.port):
        path = parsed_url.path
        if not path.endswith('/'):
            path += '/'
        if self._parsed_url.path == path:
            return self
        transport = self.__class__(other_base, _from_transport=self)
    return transport