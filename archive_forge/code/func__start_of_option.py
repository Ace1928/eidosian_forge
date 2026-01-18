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
def _start_of_option(ctx: Context, value: str) -> bool:
    """Check if the value looks like the start of an option."""
    if not value:
        return False
    c = value[0]
    return c in ctx._opt_prefixes