import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
@functools.lru_cache(maxsize=None)
def _get_ansi_pattern() -> re.Pattern:
    return re.compile('\\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])')