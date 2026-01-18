import os
import re
import typing as t
from gettext import gettext as _
from .core import Argument
from .core import BaseCommand
from .core import Context
from .core import MultiCommand
from .core import Option
from .core import Parameter
from .core import ParameterSource
from .parser import split_arg_string
from .utils import echo
def _resolve_incomplete(ctx: Context, args: t.List[str], incomplete: str) -> t.Tuple[t.Union[BaseCommand, Parameter], str]:
    """Find the Click object that will handle the completion of the
    incomplete value. Return the object and the incomplete value.

    :param ctx: Invocation context for the command represented by
        the parsed complete args.
    :param args: List of complete args before the incomplete value.
    :param incomplete: Value being completed. May be empty.
    """
    if incomplete == '=':
        incomplete = ''
    elif '=' in incomplete and _start_of_option(ctx, incomplete):
        name, _, incomplete = incomplete.partition('=')
        args.append(name)
    if '--' not in args and _start_of_option(ctx, incomplete):
        return (ctx.command, incomplete)
    params = ctx.command.get_params(ctx)
    for param in params:
        if _is_incomplete_option(ctx, args, param):
            return (param, incomplete)
    for param in params:
        if _is_incomplete_argument(ctx, param):
            return (param, incomplete)
    return (ctx.command, incomplete)