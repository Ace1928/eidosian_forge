import typing as t
from gettext import gettext as _
from gettext import ngettext
from ._compat import get_text_stderr
from .utils import echo
from .utils import format_filename
class BadArgumentUsage(UsageError):
    """Raised if an argument is generally supplied but the use of the argument
    was incorrect.  This is for instance raised if the number of values
    for an argument is not correct.

    .. versionadded:: 6.0
    """