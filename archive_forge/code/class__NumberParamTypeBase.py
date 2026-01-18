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
class _NumberParamTypeBase(ParamType):
    _number_class: t.ClassVar[t.Type[t.Any]]

    def convert(self, value: t.Any, param: t.Optional['Parameter'], ctx: t.Optional['Context']) -> t.Any:
        try:
            return self._number_class(value)
        except ValueError:
            self.fail(_('{value!r} is not a valid {number_type}.').format(value=value, number_type=self.name), param, ctx)