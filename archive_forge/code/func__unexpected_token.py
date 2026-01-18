from __future__ import annotations
import fnmatch as _fnmatch
import functools
import io
import logging
import os
import platform
import re
import sys
import textwrap
import tokenize
from typing import NamedTuple
from typing import Pattern
from typing import Sequence
from flake8 import exceptions
def _unexpected_token() -> exceptions.ExecutionError:
    return exceptions.ExecutionError(f'Expected `per-file-ignores` to be a mapping from file exclude patterns to ignore codes.\n\nConfigured `per-file-ignores` setting:\n\n{textwrap.indent(value.strip(), '    ')}')