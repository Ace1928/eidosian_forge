from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
class OperatingSystem(object):
    """An enum representing the operating system you are running on."""

    class _OS(object):
        """A single operating system."""

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

        @property
        def version(self):
            """Returns the operating system version."""
            if self == OperatingSystem.WINDOWS:
                return platform.version()
            return platform.release()

        @property
        def clean_version(self):
            """Returns a cleaned version of the operating system version."""
            version = self.version
            if self == OperatingSystem.WINDOWS:
                capitalized = version.upper()
                if capitalized in ('XP', 'VISTA'):
                    return version
                if capitalized.startswith('SERVER'):
                    return version[:11].replace(' ', '_')
            matches = re.match('(\\d+)(\\.\\d+)?(\\.\\d+)?.*', version)
            if not matches:
                return None
            return ''.join((group for group in matches.groups() if group))
    WINDOWS = _OS('WINDOWS', 'Windows', 'windows')
    MACOSX = _OS('MACOSX', 'Mac OS X', 'darwin')
    LINUX = _OS('LINUX', 'Linux', 'linux')
    CYGWIN = _OS('CYGWIN', 'Cygwin', 'cygwin')
    MSYS = _OS('MSYS', 'Msys', 'msys')
    _ALL = [WINDOWS, MACOSX, LINUX, CYGWIN, MSYS]

    @staticmethod
    def AllValues():
        """Gets all possible enum values.

    Returns:
      list, All the enum values.
    """
        return list(OperatingSystem._ALL)

    @staticmethod
    def FromId(os_id, error_on_unknown=True):
        """Gets the enum corresponding to the given operating system id.

    Args:
      os_id: str, The operating system id to parse
      error_on_unknown: bool, True to raise an exception if the id is unknown,
        False to just return None.

    Raises:
      InvalidEnumValue: If the given value cannot be parsed.

    Returns:
      OperatingSystemTuple, One of the OperatingSystem constants or None if the
      input is None.
    """
        if not os_id:
            return None
        for operating_system in OperatingSystem._ALL:
            if operating_system.id == os_id:
                return operating_system
        if error_on_unknown:
            raise InvalidEnumValue(os_id, 'Operating System', [value.id for value in OperatingSystem._ALL])
        return None

    @staticmethod
    def Current():
        """Determines the current operating system.

    Returns:
      OperatingSystemTuple, One of the OperatingSystem constants or None if it
      cannot be determined.
    """
        if os.name == 'nt':
            return OperatingSystem.WINDOWS
        elif 'linux' in sys.platform:
            return OperatingSystem.LINUX
        elif 'darwin' in sys.platform:
            return OperatingSystem.MACOSX
        elif 'cygwin' in sys.platform:
            return OperatingSystem.CYGWIN
        return None

    @staticmethod
    def IsWindows():
        """Returns True if the current operating system is Windows."""
        return OperatingSystem.Current() is OperatingSystem.WINDOWS