import errno
import itertools
import logging
import os.path
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import (
from pip._internal.utils.misc import enum, rmtree
@classmethod
def _generate_names(cls, name: str) -> Generator[str, None, None]:
    """Generates a series of temporary names.

        The algorithm replaces the leading characters in the name
        with ones that are valid filesystem characters, but are not
        valid package names (for both Python and pip definitions of
        package).
        """
    for i in range(1, len(name)):
        for candidate in itertools.combinations_with_replacement(cls.LEADING_CHARS, i - 1):
            new_name = '~' + ''.join(candidate) + name[i:]
            if new_name != name:
                yield new_name
    for i in range(len(cls.LEADING_CHARS)):
        for candidate in itertools.combinations_with_replacement(cls.LEADING_CHARS, i):
            new_name = '~' + ''.join(candidate) + name
            if new_name != name:
                yield new_name