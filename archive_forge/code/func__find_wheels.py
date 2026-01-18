import os
import textwrap
from optparse import Values
from typing import Any, List
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, PipError
from pip._internal.utils import filesystem
from pip._internal.utils.logging import getLogger
def _find_wheels(self, options: Values, pattern: str) -> List[str]:
    wheel_dir = self._cache_dir(options, 'wheels')
    pattern = pattern + ('*.whl' if '-' in pattern else '-*.whl')
    return filesystem.find_files(wheel_dir, pattern)