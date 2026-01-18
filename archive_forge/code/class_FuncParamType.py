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
class FuncParamType(ParamType):

    def __init__(self, func: t.Callable[[t.Any], t.Any]) -> None:
        self.name: str = func.__name__
        self.func = func

    def to_info_dict(self) -> t.Dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict['func'] = self.func
        return info_dict

    def convert(self, value: t.Any, param: t.Optional['Parameter'], ctx: t.Optional['Context']) -> t.Any:
        try:
            return self.func(value)
        except ValueError:
            try:
                value = str(value)
            except UnicodeError:
                value = value.decode('utf-8', 'replace')
            self.fail(value, param, ctx)