from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
class Cmd2ArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser class that improves error and help output"""

    def __init__(self, prog: Optional[str]=None, usage: Optional[str]=None, description: Optional[str]=None, epilog: Optional[str]=None, parents: Sequence[argparse.ArgumentParser]=(), formatter_class: Type[argparse.HelpFormatter]=Cmd2HelpFormatter, prefix_chars: str='-', fromfile_prefix_chars: Optional[str]=None, argument_default: Optional[str]=None, conflict_handler: str='error', add_help: bool=True, allow_abbrev: bool=True, *, ap_completer_type: Optional[Type['ArgparseCompleter']]=None) -> None:
        """
        # Custom parameter added by cmd2

        :param ap_completer_type: optional parameter which specifies a subclass of ArgparseCompleter for custom tab completion
                                  behavior on this parser. If this is None or not present, then cmd2 will use
                                  argparse_completer.DEFAULT_AP_COMPLETER when tab completing this parser's arguments
        """
        super(Cmd2ArgumentParser, self).__init__(prog=prog, usage=usage, description=description, epilog=epilog, parents=parents if parents else [], formatter_class=formatter_class, prefix_chars=prefix_chars, fromfile_prefix_chars=fromfile_prefix_chars, argument_default=argument_default, conflict_handler=conflict_handler, add_help=add_help, allow_abbrev=allow_abbrev)
        self.set_ap_completer_type(ap_completer_type)

    def add_subparsers(self, **kwargs: Any) -> argparse._SubParsersAction:
        """
        Custom override. Sets a default title if one was not given.

        :param kwargs: additional keyword arguments
        :return: argparse Subparser Action
        """
        if 'title' not in kwargs:
            kwargs['title'] = 'subcommands'
        return super().add_subparsers(**kwargs)

    def error(self, message: str) -> NoReturn:
        """Custom override that applies custom formatting to the error message"""
        lines = message.split('\n')
        linum = 0
        formatted_message = ''
        for line in lines:
            if linum == 0:
                formatted_message = 'Error: ' + line
            else:
                formatted_message += '\n       ' + line
            linum += 1
        self.print_usage(sys.stderr)
        formatted_message = ansi.style_error(formatted_message)
        self.exit(2, f'{formatted_message}\n\n')

    def format_help(self) -> str:
        """Copy of format_help() from argparse.ArgumentParser with tweaks to separately display required parameters"""
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        for action_group in self._action_groups:
            if sys.version_info >= (3, 10):
                default_options_group = action_group.title == 'options'
            else:
                default_options_group = action_group.title == 'optional arguments'
            if default_options_group:
                req_args = []
                opt_args = []
                for action in action_group._group_actions:
                    if action.required:
                        req_args.append(action)
                    else:
                        opt_args.append(action)
                formatter.start_section('required arguments')
                formatter.add_text(action_group.description)
                formatter.add_arguments(req_args)
                formatter.end_section()
                formatter.start_section('optional arguments')
                formatter.add_text(action_group.description)
                formatter.add_arguments(opt_args)
                formatter.end_section()
            else:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help() + '\n'

    def _print_message(self, message: str, file: Optional[IO[str]]=None) -> None:
        if message:
            if file is None:
                file = sys.stderr
            ansi.style_aware_write(file, message)