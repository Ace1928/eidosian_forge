import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
def _set_parser_prog(parser: argparse.ArgumentParser, prog: str) -> None:
    """
    Recursively set prog attribute of a parser and all of its subparsers so that the root command
    is a command name and not sys.argv[0].

    :param parser: the parser being edited
    :param prog: new value for the parser's prog attribute
    """
    parser.prog = prog
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            action._prog_prefix = parser.prog
            processed_parsers = []
            for subcmd_name, subcmd_parser in action.choices.items():
                if subcmd_parser in processed_parsers:
                    continue
                subcmd_prog = parser.prog + ' ' + subcmd_name
                _set_parser_prog(subcmd_parser, subcmd_prog)
                processed_parsers.append(subcmd_parser)
            break