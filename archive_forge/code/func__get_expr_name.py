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
def _get_expr_name(node):
    """Function to get either ``attrname`` or ``name`` from ``node.func.expr``

    Created specifically for the case of ``display.deprecated`` or ``self._display.deprecated``
    """
    try:
        return node.func.expr.attrname
    except AttributeError:
        return node.func.expr.name