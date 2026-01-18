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
class StringParamType(ParamType):
    name = 'text'

    def convert(self, value: t.Any, param: t.Optional['Parameter'], ctx: t.Optional['Context']) -> t.Any:
        if isinstance(value, bytes):
            enc = _get_argv_encoding()
            try:
                value = value.decode(enc)
            except UnicodeError:
                fs_enc = sys.getfilesystemencoding()
                if fs_enc != enc:
                    try:
                        value = value.decode(fs_enc)
                    except UnicodeError:
                        value = value.decode('utf-8', 'replace')
                else:
                    value = value.decode('utf-8', 'replace')
            return value
        return str(value)

    def __repr__(self) -> str:
        return 'STRING'