import typing as t
from gettext import gettext as _
from gettext import ngettext
from ._compat import get_text_stderr
from .utils import echo
from .utils import format_filename
class FileError(ClickException):
    """Raised if a file cannot be opened."""

    def __init__(self, filename: str, hint: t.Optional[str]=None) -> None:
        if hint is None:
            hint = _('unknown error')
        super().__init__(hint)
        self.ui_filename: str = format_filename(filename)
        self.filename = filename

    def format_message(self) -> str:
        return _('Could not open file {filename!r}: {message}').format(filename=self.ui_filename, message=self.message)