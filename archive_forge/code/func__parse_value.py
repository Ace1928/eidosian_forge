import argparse
import functools
import os
import pathlib
import re
import sys
import textwrap
from types import ModuleType
from typing import (
from requests.structures import CaseInsensitiveDict
import gitlab.config
from gitlab.base import RESTObject
def _parse_value(v: Any) -> Any:
    if isinstance(v, str) and v.startswith('@@'):
        return v[1:]
    if isinstance(v, str) and v.startswith('@'):
        filepath = pathlib.Path(v[1:]).expanduser().resolve()
        try:
            with open(filepath, encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'rb') as f:
                return f.read()
        except OSError as exc:
            exc_name = type(exc).__name__
            sys.stderr.write(f'{exc_name}: {exc}\n')
            sys.exit(1)
    return v