import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
def _clean_str(string: str) -> T.Optional[str]:
    string = string.strip()
    if len(string) > 0:
        return string
    return None