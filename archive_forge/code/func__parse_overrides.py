import hashlib
import os
import shutil
import sys
from configparser import InterpolationError
from contextlib import contextmanager
from pathlib import Path
from typing import (
import srsly
import typer
from click import NoSuchOption
from click.parser import split_arg_string
from thinc.api import Config, ConfigValidationError, require_gpu
from thinc.util import gpu_is_available
from typer.main import get_command
from wasabi import Printer, msg
from weasel import app as project_cli
from .. import about
from ..compat import Literal
from ..schemas import validate
from ..util import (
def _parse_overrides(args: List[str], is_cli: bool=False) -> Dict[str, Any]:
    result = {}
    while args:
        opt = args.pop(0)
        err = f"Invalid config override '{opt}'"
        if opt.startswith('--'):
            orig_opt = opt
            opt = opt.replace('--', '')
            if '.' not in opt:
                if is_cli:
                    raise NoSuchOption(orig_opt)
                else:
                    msg.fail(f"{err}: can't override top-level sections", exits=1)
            if '=' in opt:
                opt, value = opt.split('=', 1)
                opt = opt.replace('-', '_')
            elif not args or args[0].startswith('--'):
                value = 'true'
            else:
                value = args.pop(0)
            result[opt] = _parse_override(value)
        else:
            msg.fail(f'{err}: name should start with --', exits=1)
    return result