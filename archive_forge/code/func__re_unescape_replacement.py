import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
def _re_unescape_replacement(match: Match[str]) -> str:
    group = match.group(1)
    if group[0] in _alphanum:
        raise ValueError("cannot unescape '\\\\%s'" % group[0])
    return group