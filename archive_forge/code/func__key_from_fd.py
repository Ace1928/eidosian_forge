from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
def _key_from_fd(self, fd):
    """Return the key associated to a given file descriptor.

        Parameters:
        fd -- file descriptor

        Returns:
        corresponding key, or None if not found
        """
    try:
        return self._fd_to_key[fd]
    except KeyError:
        return None