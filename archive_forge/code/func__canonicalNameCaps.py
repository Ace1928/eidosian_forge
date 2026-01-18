from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def _canonicalNameCaps(self, name: bytes) -> bytes:
    """
        Return the canonical name for the given header.

        @param name: The all-lowercase header name to capitalize in its
            canonical form.

        @return: The canonical name of the header.
        """
    return self._caseMappings.get(name, _dashCapitalize(name))