import errno
import inspect
import os
import sys
from enum import Enum
from gettext import gettext as _
from typing import (
import click
import click.core
import click.formatting
import click.parser
import click.shell_completion
import click.types
import click.utils
def _split_opt(opt: str) -> Tuple[str, str]:
    first = opt[:1]
    if first.isalnum():
        return ('', opt)
    if opt[1:2] == first:
        return (opt[:2], opt[2:])
    return (first, opt[1:])