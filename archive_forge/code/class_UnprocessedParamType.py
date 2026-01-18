import os
import stat
import sys
import typing as t
from datetime import datetime
from gettext import gettext as _
from gettext import ngettext
from ._compat import _get_argv_encoding
from ._compat import open_stream
from .exceptions import BadParameter
from .utils import format_filename
from .utils import LazyFile
from .utils import safecall
class UnprocessedParamType(ParamType):
    name = 'text'

    def convert(self, value: t.Any, param: t.Optional['Parameter'], ctx: t.Optional['Context']) -> t.Any:
        return value

    def __repr__(self) -> str:
        return 'UNPROCESSED'