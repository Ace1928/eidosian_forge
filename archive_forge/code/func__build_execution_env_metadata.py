import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_execution_env_metadata(self):
    """
        Build the execution environment component of the User-Agent header.

        Returns a single component prefixed with "exec-env", usually sourced
        from the environment variable AWS_EXECUTION_ENV.
        """
    if self._execution_env:
        return [UserAgentComponent('exec-env', self._execution_env)]
    else:
        return []