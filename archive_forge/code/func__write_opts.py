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
def _write_opts(opts: Sequence[str]) -> str:
    nonlocal any_prefix_is_slash
    rv, any_slashes = click.formatting.join_options(opts)
    if any_slashes:
        any_prefix_is_slash = True
    if not self.is_flag and (not self.count):
        rv += f' {self.make_metavar()}'
    return rv