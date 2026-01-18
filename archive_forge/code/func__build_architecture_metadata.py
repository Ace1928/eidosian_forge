import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_architecture_metadata(self):
    """
        Build architecture component of the User-Agent header string.

        Returns the machine type with prefix "md" and name "arch", if one is
        available. Common values include "x86_64", "arm64", "i386".
        """
    if self._platform_machine:
        return [UserAgentComponent('md', 'arch', self._platform_machine.lower())]
    return []