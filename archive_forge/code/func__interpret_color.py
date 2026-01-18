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
def _interpret_color(color: t.Union[int, t.Tuple[int, int, int], str], offset: int=0) -> str:
    if isinstance(color, int):
        return f'{38 + offset};5;{color:d}'
    if isinstance(color, (tuple, list)):
        r, g, b = color
        return f'{38 + offset};2;{r:d};{g:d};{b:d}'
    return str(_ansi_colors[color] + offset)