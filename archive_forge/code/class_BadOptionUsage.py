import typing as t
from gettext import gettext as _
from gettext import ngettext
from ._compat import get_text_stderr
from .utils import echo
from .utils import format_filename
class BadOptionUsage(UsageError):
    """Raised if an option is generally supplied but the use of the option
    was incorrect.  This is for instance raised if the number of arguments
    for an option is not correct.

    .. versionadded:: 4.0

    :param option_name: the name of the option being used incorrectly.
    """

    def __init__(self, option_name: str, message: str, ctx: t.Optional['Context']=None) -> None:
        super().__init__(message, ctx)
        self.option_name = option_name