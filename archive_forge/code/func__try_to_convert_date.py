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
def _try_to_convert_date(self, value: t.Any, format: str) -> t.Optional[datetime]:
    try:
        return datetime.strptime(value, format)
    except ValueError:
        return None