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
def _is_incomplete_argument(ctx: Context, param: Parameter) -> bool:
    """Determine if the given parameter is an argument that can still
    accept values.

    :param ctx: Invocation context for the command represented by the
        parsed complete args.
    :param param: Argument object being checked.
    """
    if not isinstance(param, Argument):
        return False
    assert param.name is not None
    value = ctx.params.get(param.name)
    return param.nargs == -1 or ctx.get_parameter_source(param.name) is not ParameterSource.COMMANDLINE or (param.nargs > 1 and isinstance(value, (tuple, list)) and (len(value) < param.nargs))