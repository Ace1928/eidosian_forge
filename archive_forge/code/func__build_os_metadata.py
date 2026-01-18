import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_os_metadata(self):
    """
        Build the OS/platform components of the User-Agent header string.

        For recognized platform names that match or map to an entry in the list
        of standardized OS names, a single component with prefix "os" is
        returned. Otherwise, one component "os/other" is returned and a second
        with prefix "md" and the raw platform name.

        String representations of example return values:
         * ``os/macos#10.13.6``
         * ``os/linux``
         * ``os/other``
         * ``os/other md/foobar#1.2.3``
        """
    if self._platform_name is None:
        return [UserAgentComponent('os', 'other')]
    plt_name_lower = self._platform_name.lower()
    if plt_name_lower in _USERAGENT_ALLOWED_OS_NAMES:
        os_family = plt_name_lower
    elif plt_name_lower in _USERAGENT_PLATFORM_NAME_MAPPINGS:
        os_family = _USERAGENT_PLATFORM_NAME_MAPPINGS[plt_name_lower]
    else:
        os_family = None
    if os_family is not None:
        return [UserAgentComponent('os', os_family, self._platform_version)]
    else:
        return [UserAgentComponent('os', 'other'), UserAgentComponent('md', self._platform_name, self._platform_version)]