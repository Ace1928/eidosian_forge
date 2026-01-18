import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
def _parse_positionals(args: Tuple[Any, ...]) -> Tuple['cmd2.Cmd', Union[Statement, str]]:
    """
    Helper function for cmd2 decorators to inspect the positional arguments until the cmd2.Cmd argument is found
    Assumes that we will find cmd2.Cmd followed by the command statement object or string.
    :arg args: The positional arguments to inspect
    :return: The cmd2.Cmd reference and the command line statement
    """
    for pos, arg in enumerate(args):
        from cmd2 import Cmd
        if (isinstance(arg, Cmd) or isinstance(arg, CommandSet)) and len(args) > pos:
            if isinstance(arg, CommandSet):
                arg = arg._cmd
            next_arg = args[pos + 1]
            if isinstance(next_arg, (Statement, str)):
                return (arg, args[pos + 1])
    raise TypeError('Expected arguments: cmd: cmd2.Cmd, statement: Union[Statement, str] Not found')