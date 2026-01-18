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
class TyroArgumentParser(argparse.ArgumentParser):
    _parser_specification: ParserSpecification
    _parsing_known_args: bool
    _args: List[str]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def _parse_known_args(self, arg_strings, namespace):
        """We override _parse_known_args() to improve error messages in the presence of
        subcommands. Difference is marked with <new>...</new> below."""
        if ' ' not in self.prog:
            global global_unrecognized_args
            global_unrecognized_args = []
        if self.fromfile_prefix_chars is not None:
            arg_strings = self._read_args_from_files(arg_strings)
        action_conflicts = {}
        for mutex_group in self._mutually_exclusive_groups:
            group_actions = mutex_group._group_actions
            for i, mutex_action in enumerate(mutex_group._group_actions):
                conflicts = action_conflicts.setdefault(mutex_action, [])
                conflicts.extend(group_actions[:i])
                conflicts.extend(group_actions[i + 1:])
        option_string_indices = {}
        arg_string_pattern_parts = []
        arg_strings_iter = iter(arg_strings)
        for i, arg_string in enumerate(arg_strings_iter):
            if arg_string == '--':
                arg_string_pattern_parts.append('-')
                for arg_string in arg_strings_iter:
                    arg_string_pattern_parts.append('A')
            else:
                option_tuple = self._parse_optional(arg_string)
                if option_tuple is None:
                    pattern = 'A'
                else:
                    option_string_indices[i] = option_tuple
                    pattern = 'O'
                arg_string_pattern_parts.append(pattern)
        arg_strings_pattern = ''.join(arg_string_pattern_parts)
        seen_actions = set()
        seen_non_default_actions = set()

        def take_action(action, argument_strings, option_string=None):
            seen_actions.add(action)
            argument_values = self._get_values(action, argument_strings)
            if argument_values is not action.default:
                seen_non_default_actions.add(action)
                for conflict_action in action_conflicts.get(action, []):
                    if conflict_action in seen_non_default_actions:
                        msg = _('not allowed with argument %s')
                        action_name = argparse._get_action_name(conflict_action)
                        raise argparse.ArgumentError(action, msg % action_name)
            if argument_values is not argparse.SUPPRESS:
                action(self, namespace, argument_values, option_string)

        def consume_optional(start_index):
            option_tuple = option_string_indices[start_index]
            action, option_string, explicit_arg = option_tuple
            match_argument = self._match_argument
            action_tuples = []
            while True:
                if action is None:
                    if not self._parsing_known_args:
                        global_unrecognized_args.append(option_string)
                    extras.append(arg_strings[start_index])
                    return start_index + 1
                if explicit_arg is not None:
                    arg_count = match_argument(action, 'A')
                    chars = self.prefix_chars
                    if arg_count == 0 and option_string[1] not in chars and (explicit_arg != ''):
                        action_tuples.append((action, [], option_string))
                        char = option_string[0]
                        option_string = char + explicit_arg[0]
                        new_explicit_arg = explicit_arg[1:] or None
                        optionals_map = self._option_string_actions
                        if option_string in optionals_map:
                            action = optionals_map[option_string]
                            explicit_arg = new_explicit_arg
                        else:
                            msg = _('ignored explicit argument %r')
                            raise argparse.ArgumentError(action, msg % explicit_arg)
                    elif arg_count == 1:
                        stop = start_index + 1
                        args = [explicit_arg]
                        action_tuples.append((action, args, option_string))
                        break
                    else:
                        msg = _('ignored explicit argument %r')
                        raise argparse.ArgumentError(action, msg % explicit_arg)
                else:
                    start = start_index + 1
                    selected_patterns = arg_strings_pattern[start:]
                    arg_count = match_argument(action, selected_patterns)
                    stop = start + arg_count
                    args = arg_strings[start:stop]
                    action_tuples.append((action, args, option_string))
                    break
            assert action_tuples
            for action, args, option_string in action_tuples:
                take_action(action, args, option_string)
            return stop
        positionals = self._get_positional_actions()

        def consume_positionals(start_index):
            match_partial = self._match_arguments_partial
            selected_pattern = arg_strings_pattern[start_index:]
            arg_counts = match_partial(positionals, selected_pattern)
            for action, arg_count in zip(positionals, arg_counts):
                args = arg_strings[start_index:start_index + arg_count]
                start_index += arg_count
                take_action(action, args)
            positionals[:] = positionals[len(arg_counts):]
            return start_index
        extras = []
        start_index = 0
        if option_string_indices:
            max_option_string_index = max(option_string_indices)
        else:
            max_option_string_index = -1
        while start_index <= max_option_string_index:
            next_option_string_index = min([index for index in option_string_indices if index >= start_index])
            if start_index != next_option_string_index:
                positionals_end_index = consume_positionals(start_index)
                if positionals_end_index > start_index:
                    start_index = positionals_end_index
                    continue
                else:
                    start_index = positionals_end_index
            if start_index not in option_string_indices:
                strings = arg_strings[start_index:next_option_string_index]
                extras.extend(strings)
                start_index = next_option_string_index
            start_index = consume_optional(start_index)
        stop_index = consume_positionals(start_index)
        extras.extend(arg_strings[stop_index:])
        required_actions = []
        for action in self._actions:
            if action not in seen_actions:
                if action.required:
                    required_actions.append(argparse._get_action_name(action))
                elif action.default is not None and isinstance(action.default, str) and hasattr(namespace, action.dest) and (action.default is getattr(namespace, action.dest)):
                    setattr(namespace, action.dest, self._get_value(action, action.default))
        if required_actions:
            self.error(_('the following arguments are required: %s') % ', '.join(required_actions))
        for group in self._mutually_exclusive_groups:
            if group.required:
                for action in group._group_actions:
                    if action in seen_non_default_actions:
                        break
                else:
                    names = [argparse._get_action_name(action) for action in group._group_actions if action.help is not argparse.SUPPRESS]
                    msg = _('one of the arguments %s is required')
                    self.error(msg % ' '.join(names))
        return (namespace, extras)

    @override
    def error(self, message: str) -> NoReturn:
        """Improve error messages from argparse.

        error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        console = Console(theme=THEME.as_rich_theme())
        extra_info: List[RenderableType] = []
        global global_unrecognized_args
        if len(global_unrecognized_args) == 0 and message.startswith('unrecognized options: '):
            global_unrecognized_args = message.partition(':')[2].strip().split(' ')
        message_title = 'Parsing error'
        if len(global_unrecognized_args) > 0:
            message_title = 'Unrecognized options'
            message = f'Unrecognized options: {' '.join(global_unrecognized_args)}'
            unrecognized_arguments = set((arg for arg in global_unrecognized_args if arg.startswith('--')))
            arguments, has_subcommands, same_exists = recursive_arg_search(args=self._args, parser_spec=self._parser_specification, prog=self.prog.partition(' ')[0], unrecognized_arguments=unrecognized_arguments)
            if has_subcommands and same_exists:
                message = f'Unrecognized or misplaced options: {' '.join(global_unrecognized_args)}'
            for unrecognized_argument in unrecognized_arguments:
                scored_arguments: List[Tuple[_ArgumentInfo, float]] = []
                for arg_info in arguments:
                    assert unrecognized_argument.startswith('--')

                    def get_score(option_string: str) -> float:
                        if option_string.endswith(unrecognized_argument[2:]) or option_string.startswith(unrecognized_argument[2:]):
                            return 0.9
                        elif len(unrecognized_argument) >= 4 and all(map(lambda part: part in option_string, unrecognized_argument[2:].split('.'))):
                            return 0.9
                        else:
                            return difflib.SequenceMatcher(a=unrecognized_argument, b=option_string).ratio()
                    scored_arguments.append((arg_info, max(map(get_score, arg_info.option_strings))))
                prev_arg_option_strings: Optional[Tuple[str, ...]] = None
                show_arguments: List[_ArgumentInfo] = []
                unique_counter = 0
                for arg_info, score in sorted(scored_arguments, key=lambda arg_score: (-arg_score[1], -arg_score[0].subcommand_match_score, arg_score[0].option_strings[0], arg_score[0].usage_hint, arg_score[0].help)):
                    if score < 0.8:
                        break
                    if score < 0.9 and unique_counter >= 3 and (prev_arg_option_strings != arg_info.option_strings):
                        break
                    unique_counter += prev_arg_option_strings != arg_info.option_strings
                    show_arguments.append(arg_info)
                    prev_arg_option_strings = arg_info.option_strings
                prev_arg_option_strings = None
                prev_argument_help: Optional[str] = None
                same_counter = 0
                dots_printed = False
                if len(show_arguments) > 0:
                    extra_info.append(Rule(style=Style(color='red')))
                    extra_info.append('Perhaps you meant:' if len(unrecognized_arguments) == 1 else f'Arguments similar to {unrecognized_argument}:')
                unique_counter = 0
                for arg_info in show_arguments:
                    same_counter += 1
                    if arg_info.option_strings != prev_arg_option_strings:
                        same_counter = 0
                        if unique_counter >= 10:
                            break
                        unique_counter += 1
                    if len(show_arguments) >= 8 and same_counter >= 4 and (arg_info.option_strings == prev_arg_option_strings):
                        if not dots_printed:
                            extra_info.append(Padding('[...]', (0, 0, 0, 12)))
                        dots_printed = True
                        continue
                    if not (has_subcommands and arg_info.option_strings == prev_arg_option_strings):
                        extra_info.append(Padding('[bold]' + (', '.join(arg_info.option_strings) if arg_info.metavar is None else ', '.join(arg_info.option_strings) + ' ' + arg_info.metavar) + '[/bold]', (0, 0, 0, 4)))
                    if arg_info.help is not None and (arg_info.help != prev_argument_help or arg_info.option_strings != prev_arg_option_strings):
                        extra_info.append(Padding(arg_info.help, (0, 0, 0, 8)))
                    if has_subcommands:
                        extra_info.append(Padding(f'in [green]{arg_info.usage_hint}[/green]', (0, 0, 0, 12)))
                    prev_arg_option_strings = arg_info.option_strings
                    prev_argument_help = arg_info.help
        elif message.startswith('the following arguments are required:'):
            message_title = 'Required options'
            info_from_required_arg: Dict[str, Optional[_ArgumentInfo]] = {}
            for arg in message.partition(':')[2].strip().split(', '):
                info_from_required_arg[arg] = None
            arguments, has_subcommands, same_exists = recursive_arg_search(args=self._args, parser_spec=self._parser_specification, prog=self.prog.partition(' ')[0], unrecognized_arguments=set())
            del same_exists
            for arg_info in arguments:
                for option_string in arg_info.option_strings:
                    if option_string in info_from_required_arg and (info_from_required_arg[option_string] is None or arg_info.subcommand_match_score > info_from_required_arg[option_string].subcommand_match_score):
                        info_from_required_arg[option_string] = arg_info
            first = True
            for maybe_arg in info_from_required_arg.values():
                if maybe_arg is None:
                    continue
                if first:
                    extra_info.extend([Rule(style=Style(color='red')), 'Argument helptext:'])
                    first = False
                extra_info.append(Padding('[bold]' + (', '.join(maybe_arg.option_strings) if maybe_arg.metavar is None else ', '.join(maybe_arg.option_strings) + ' ' + maybe_arg.metavar) + '[/bold]', (0, 0, 0, 4)))
                if maybe_arg.help is not None:
                    extra_info.append(Padding(maybe_arg.help, (0, 0, 0, 8)))
                if has_subcommands:
                    extra_info.append(Padding(f'in [green]{maybe_arg.usage_hint}[/green]', (0, 0, 0, 12)))
        console.print(Panel(Group(f'{message[0].upper() + message[1:]}' if len(message) > 0 else '', *extra_info, Rule(style=Style(color='red')), f'For full helptext, run [bold]{self.prog} --help[/bold]'), title=f'[bold]{message_title}[/bold]', title_align='left', border_style=Style(color='bright_red'), expand=False))
        sys.exit(2)