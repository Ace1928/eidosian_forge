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
def _extract_default_help_str(self, *, ctx: click.Context) -> Optional[Union[Any, Callable[[], Any]]]:
    return _extract_default_help_str(self, ctx=ctx)