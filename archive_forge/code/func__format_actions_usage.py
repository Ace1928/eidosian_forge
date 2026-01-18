from __future__ import annotations
import argparse
import contextlib
import dataclasses
import difflib
import itertools
import re as _re
import shlex
import shutil
import sys
from gettext import gettext as _
from typing import Any, Dict, Generator, Iterable, List, NoReturn, Optional, Set, Tuple
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from typing_extensions import override
from . import _arguments, _strings, conf
from ._parsers import ParserSpecification
def _format_actions_usage(self, actions, groups):
    """Backporting from Python 3.10, primarily to call format_usage() on actions."""
    group_actions = set()
    inserts = {}
    for group in groups:
        if not group._group_actions:
            raise ValueError(f'empty group {group}')
        try:
            start = actions.index(group._group_actions[0])
        except ValueError:
            continue
        else:
            group_action_count = len(group._group_actions)
            end = start + group_action_count
            if actions[start:end] == group._group_actions:
                suppressed_actions_count = 0
                for action in group._group_actions:
                    group_actions.add(action)
                    if action.help is argparse.SUPPRESS:
                        suppressed_actions_count += 1
                exposed_actions_count = group_action_count - suppressed_actions_count
                if not group.required:
                    if start in inserts:
                        inserts[start] += ' ['
                    else:
                        inserts[start] = '['
                    if end in inserts:
                        inserts[end] += ']'
                    else:
                        inserts[end] = ']'
                elif exposed_actions_count > 1:
                    if start in inserts:
                        inserts[start] += ' ('
                    else:
                        inserts[start] = '('
                    if end in inserts:
                        inserts[end] += ')'
                    else:
                        inserts[end] = ')'
                for i in range(start + 1, end):
                    inserts[i] = '|'
    parts = []
    for i, action in enumerate(actions):
        if action.help is argparse.SUPPRESS:
            parts.append(None)
            if inserts.get(i) == '|':
                inserts.pop(i)
            elif inserts.get(i + 1) == '|':
                inserts.pop(i + 1)
        elif not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            part = self._format_args(action, default)
            if action in group_actions:
                if part[0] == '[' and part[-1] == ']':
                    part = part[1:-1]
            parts.append(part)
        else:
            option_string = action.option_strings[0]
            if action.nargs == 0:
                part = action.format_usage() if hasattr(action, 'format_usage') else '%s' % option_string
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                part = '%s %s' % (option_string, args_string)
            if not action.required and action not in group_actions:
                part = '[%s]' % part
            parts.append(part)
    for i in sorted(inserts, reverse=True):
        parts[i:i] = [inserts[i]]
    text = ' '.join([item for item in parts if item is not None])
    open = '[\\[(]'
    close = '[\\])]'
    text = _re.sub('(%s) ' % open, '\\1', text)
    text = _re.sub(' (%s)' % close, '\\1', text)
    text = _re.sub('%s *%s' % (open, close), '', text)
    text = text.strip()
    return text