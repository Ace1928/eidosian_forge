from __future__ import annotations
import datetime
import functools
import json
import re
import shlex
import typing as t
from tokenize import COMMENT, TokenInfo
import astroid
from pylint.checkers import BaseChecker, BaseTokenChecker
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.six import string_types
from ansible.release import __version__ as ansible_version_raw
from ansible.utils.version import SemanticVersion
@functools.cached_property
def _min_python_version_db(self) -> dict[str, str]:
    """A dictionary of absolute file paths and their minimum required Python version."""
    with open(self.linter.config.min_python_version_db) as db_file:
        return json.load(db_file)