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
def _check_date(self, node, date):
    if not isinstance(date, str):
        self.add_message('ansible-invalid-deprecated-date', node=node, args=(date,))
        return
    try:
        date_parsed = parse_isodate(date)
    except ValueError:
        self.add_message('ansible-invalid-deprecated-date', node=node, args=(date,))
        return
    if date_parsed < datetime.date.today():
        self.add_message('ansible-deprecated-date', node=node, args=(date,))