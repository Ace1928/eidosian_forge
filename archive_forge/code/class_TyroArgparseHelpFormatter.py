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
class TyroArgparseHelpFormatter(argparse.RawDescriptionHelpFormatter):

    def __init__(self, prog: str):
        indent_increment = 4
        width = shutil.get_terminal_size().columns - 2
        max_help_position = 24
        self._fixed_help_position = False
        self._strip_ansi_sequences = not _arguments.USE_RICH
        super().__init__(prog, indent_increment, max_help_position, width)

    @override
    def _format_args(self, action, default_metavar):
        """Override _format_args() to ignore nargs and always expect single string
        metavars."""
        get_metavar = self._metavar_formatter(action, default_metavar)
        out = get_metavar(1)[0]
        if isinstance(out, str):
            return out if self._strip_ansi_sequences else str_from_rich(Text.from_ansi(out, style=THEME.metavar_fixed if out == '{fixed}' else THEME.metavar), soft_wrap=True)
        return out

    @override
    def add_argument(self, action):
        prev_max_length = self._action_max_length
        super().add_argument(action)
        if self._action_max_length > self._max_help_position + 2:
            self._action_max_length = prev_max_length

    def _split_lines(self, text, width):
        text = self._whitespace_matcher.sub(' ', text).strip()
        import textwrap as textwrap
        textwrap.len = monkeypatch_len
        out = textwrap.wrap(text, width)
        del textwrap.len
        return out

    @override
    def _fill_text(self, text, width, indent):
        return ''.join((indent + line for line in text.splitlines(keepends=True)))

    @override
    def format_help(self):
        self._tyro_rule = None
        self._fixed_help_position = False
        help1 = super().format_help()
        self._tyro_rule = None
        self._fixed_help_position = True
        help2 = super().format_help()
        out = help1 if help1.count('\n') < help2.count('\n') else help2
        if self._strip_ansi_sequences:
            return _strings.strip_ansi_sequences(out)
        else:
            return out

    @override
    class _Section(object):

        def __init__(self, formatter, parent, heading=None):
            self.formatter = formatter
            self.parent = parent
            self.heading = heading
            self.items = []
            self.formatter._tyro_rule = None

        def format_help(self):
            if self.parent is None:
                return self._tyro_format_root()
            else:
                return self._tyro_format_nonroot()

        def _tyro_format_root(self):
            console = Console(width=self.formatter._width, theme=THEME.as_rich_theme())
            with console.capture() as capture:
                top_parts = []
                column_parts = []
                column_parts_lines = []
                for func, args in self.items:
                    item_content = func(*args)
                    if item_content is None:
                        pass
                    elif isinstance(item_content, str):
                        if item_content.strip() == '':
                            continue
                        top_parts.append(Text.from_ansi(item_content))
                    else:
                        assert isinstance(item_content, Panel)
                        column_parts.append(item_content)
                        column_parts_lines.append(str_from_rich(item_content, width=65).strip().count('\n') + 1)
                min_column_width = 65
                height_breakpoint = 50
                column_count = max(1, min(sum(column_parts_lines) // height_breakpoint + 1, self.formatter._width // min_column_width, len(column_parts)))
                if column_count > 1:
                    column_width = self.formatter._width // column_count - 1
                    column_parts_lines = map(lambda p: str_from_rich(p, width=column_width).strip().count('\n') + 1, column_parts)
                else:
                    column_width = None
                column_lines = [0 for i in range(column_count)]
                column_parts_grouped = [[] for i in range(column_count)]
                for p, l in zip(column_parts, column_parts_lines):
                    chosen_column = column_lines.index(min(column_lines))
                    column_parts_grouped[chosen_column].append(p)
                    column_lines[chosen_column] += l
                columns = Columns([Group(*g) for g in column_parts_grouped], column_first=True, width=column_width)
                console.print(Group(*top_parts))
                console.print(columns)
            return capture.get()

        def _format_action(self, action: argparse.Action):
            invocation = self.formatter._format_action_invocation(action)
            indent = self.formatter._current_indent
            help_position = min(self.formatter._action_max_length + 4, self.formatter._max_help_position)
            if self.formatter._fixed_help_position:
                help_position = 4
            item_parts: List[RenderableType] = []
            if action.option_strings == ['-h', '--help']:
                assert action.help is not None
                action.help = str_from_rich(Text.from_markup('[helptext]' + action.help + '[/helptext]'))
            if action.help is not None:
                assert isinstance(action.help, str)
                helptext = Text.from_ansi(action.help.replace('%%', '%')) if _strings.strip_ansi_sequences(action.help) != action.help else Text.from_markup(action.help.replace('%%', '%'))
            else:
                helptext = Text('')
            if action.help and len(_strings.strip_ansi_sequences(invocation)) + indent < help_position - 1 and (not self.formatter._fixed_help_position):
                table = Table(show_header=False, box=None, padding=0)
                table.add_column(width=help_position - indent)
                table.add_column()
                table.add_row(Text.from_ansi(invocation, style=THEME.invocation), helptext)
                item_parts.append(table)
            else:
                item_parts.append(Text.from_ansi(invocation + '\n', style=THEME.invocation))
                if action.help:
                    item_parts.append(Padding(helptext, pad=(0, 0, 0, help_position - indent)))
            try:
                subaction: argparse.Action
                for subaction in action._get_subactions():
                    self.formatter._indent()
                    item_parts.append(Padding(Group(*self._format_action(subaction)), pad=(0, 0, 0, self.formatter._indent_increment)))
                    self.formatter._dedent()
            except AttributeError:
                pass
            return item_parts

        def _tyro_format_nonroot(self):
            description_part = None
            item_parts = []
            for func, args in self.items:
                if getattr(func, '__func__', None) is TyroArgparseHelpFormatter._format_action:
                    action, = args
                    assert isinstance(action, argparse.Action)
                    item_parts.extend(self._format_action(action))
                else:
                    item_content = func(*args)
                    assert isinstance(item_content, str)
                    if item_content.strip() != '':
                        assert description_part is None
                        description_part = Text.from_ansi(item_content.strip() + '\n', style=THEME.description)
            if len(item_parts) == 0:
                return None
            if self.heading is not argparse.SUPPRESS and self.heading is not None:
                current_indent = self.formatter._current_indent
                heading = '%*s%s:\n' % (current_indent, '', self.heading)
                heading = heading.strip()[:-1]
            else:
                heading = ''
            lines = list(itertools.chain(*map(lambda p: _strings.strip_ansi_sequences(str_from_rich(p, width=self.formatter._width, soft_wrap=True)).rstrip().split('\n'), item_parts + [description_part] if description_part is not None else item_parts)))
            max_width = max(map(len, lines))
            if self.formatter._tyro_rule is None:
                self.formatter._tyro_rule = Text.from_ansi('─' * max_width, style=THEME.border, overflow='crop')
            elif len(self.formatter._tyro_rule._text[0]) < max_width:
                self.formatter._tyro_rule._text = ['─' * max_width]
            if description_part is not None:
                item_parts = [description_part, self.formatter._tyro_rule] + item_parts
            return Panel(Group(*item_parts), title=heading, title_align='left', border_style=THEME.border)

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

    @override
    def _format_usage(self, usage, actions: Iterable[argparse.Action], groups, prefix) -> str:
        assert isinstance(actions, list)
        if len(actions) > 4:
            new_actions = []
            prog_parts = shlex.split(self._prog)
            added_options = False
            for action in actions:
                if action.dest == 'help' or len(action.option_strings) == 0:
                    new_actions.append(action)
                elif not added_options:
                    added_options = True
                    new_actions.append(argparse.Action(['OPTIONS' if len(prog_parts) == 1 else prog_parts[-1].upper() + ' OPTIONS'], dest=''))
            actions = new_actions
        if prefix is None:
            prefix = str_from_rich('[bold]usage[/bold]: ')
        usage = super()._format_usage(usage, actions, groups, prefix)
        return '\n\n' + usage