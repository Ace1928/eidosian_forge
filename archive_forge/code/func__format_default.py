import inspect
import io
import itertools
import sys
import typing as t
from gettext import gettext as _
from ._compat import isatty
from ._compat import strip_ansi
from .exceptions import Abort
from .exceptions import UsageError
from .globals import resolve_color_default
from .types import Choice
from .types import convert_type
from .types import ParamType
from .utils import echo
from .utils import LazyFile
def _format_default(default: t.Any) -> t.Any:
    if isinstance(default, (io.IOBase, LazyFile)) and hasattr(default, 'name'):
        return default.name
    return default