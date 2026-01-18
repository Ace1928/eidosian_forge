import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _update_credentials(self, credentials):
    """Update the credentials of the current connection.

        Some protocols can renegociate the credentials within a connection,
        this method allows daughter classes to share updated credentials.

        :param credentials: the updated credentials.
        """
    self._shared_connection.credentials = credentials