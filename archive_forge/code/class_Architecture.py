from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
class Architecture(object):
    """An enum representing the system architecture you are running on."""

    class _ARCH(object):
        """A single architecture."""

        def __init__(self, id, name, file_name):
            self.id = id
            self.name = name
            self.file_name = file_name

        def __str__(self):
            return self.id

        def __eq__(self, other):
            return isinstance(other, type(self)) and self.id == other.id and (self.name == other.name) and (self.file_name == other.file_name)

        def __hash__(self):
            return hash(self.id) + hash(self.name) + hash(self.file_name)

        def __ne__(self, other):
            return not self == other

        @classmethod
        def _CmpHelper(cls, x, y):
            """Just a helper equivalent to the cmp() function in Python 2."""
            return (x > y) - (x < y)

        def __lt__(self, other):
            return self._CmpHelper((self.id, self.name, self.file_name), (other.id, other.name, other.file_name)) < 0

        def __gt__(self, other):
            return self._CmpHelper((self.id, self.name, self.file_name), (other.id, other.name, other.file_name)) > 0

        def __le__(self, other):
            return not self.__gt__(other)

        def __ge__(self, other):
            return not self.__lt__(other)
    x86 = _ARCH('x86', 'x86', 'x86')
    x86_64 = _ARCH('x86_64', 'x86_64', 'x86_64')
    ppc = _ARCH('PPC', 'PPC', 'ppc')
    arm = _ARCH('arm', 'arm', 'arm')
    _ALL = [x86, x86_64, ppc, arm]
    _MACHINE_TO_ARCHITECTURE = {'amd64': x86_64, 'x86_64': x86_64, 'i686-64': x86_64, 'i386': x86, 'i686': x86, 'x86': x86, 'ia64': x86, 'powerpc': ppc, 'power macintosh': ppc, 'ppc64': ppc, 'armv6': arm, 'armv6l': arm, 'arm64': arm, 'armv7': arm, 'armv7l': arm, 'aarch64': arm}

    @staticmethod
    def AllValues():
        """Gets all possible enum values.

    Returns:
      list, All the enum values.
    """
        return list(Architecture._ALL)

    @staticmethod
    def FromId(architecture_id, error_on_unknown=True):
        """Gets the enum corresponding to the given architecture id.

    Args:
      architecture_id: str, The architecture id to parse
      error_on_unknown: bool, True to raise an exception if the id is unknown,
        False to just return None.

    Raises:
      InvalidEnumValue: If the given value cannot be parsed.

    Returns:
      ArchitectureTuple, One of the Architecture constants or None if the input
      is None.
    """
        if not architecture_id:
            return None
        for arch in Architecture._ALL:
            if arch.id == architecture_id:
                return arch
        if error_on_unknown:
            raise InvalidEnumValue(architecture_id, 'Architecture', [value.id for value in Architecture._ALL])
        return None

    @staticmethod
    def Current():
        """Determines the current system architecture.

    Returns:
      ArchitectureTuple, One of the Architecture constants or None if it cannot
      be determined.
    """
        return Architecture._MACHINE_TO_ARCHITECTURE.get(platform.machine().lower())