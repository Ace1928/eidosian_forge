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
def _process_python_version(self, token: TokenInfo, data: dict[str, str]) -> None:
    current_file = self.linter.current_file
    check_version = self._min_python_version_db[current_file]
    try:
        if LooseVersion(data['python_version']) < LooseVersion(check_version):
            self.add_message('ansible-deprecated-python-version-comment', line=token.start[0], col_offset=token.start[1], args=(data['python_version'], data['description'] or 'description not provided'))
    except (ValueError, TypeError) as exc:
        self.add_message('ansible-deprecated-version-comment-invalid-version', line=token.start[0], col_offset=token.start[1], args=(data['python_version'], exc))