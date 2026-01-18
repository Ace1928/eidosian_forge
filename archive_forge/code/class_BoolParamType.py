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
class BoolParamType(ParamType):
    name = 'boolean'

    def convert(self, value: t.Any, param: t.Optional['Parameter'], ctx: t.Optional['Context']) -> t.Any:
        if value in {False, True}:
            return bool(value)
        norm = value.strip().lower()
        if norm in {'1', 'true', 't', 'yes', 'y', 'on'}:
            return True
        if norm in {'0', 'false', 'f', 'no', 'n', 'off'}:
            return False
        self.fail(_('{value!r} is not a valid boolean.').format(value=value), param, ctx)

    def __repr__(self) -> str:
        return 'BOOL'