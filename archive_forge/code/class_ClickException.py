import typing as t
from gettext import gettext as _
from gettext import ngettext
from ._compat import get_text_stderr
from .utils import echo
from .utils import format_filename
class ClickException(Exception):
    """An exception that Click can handle and show to the user."""
    exit_code = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def format_message(self) -> str:
        return self.message

    def __str__(self) -> str:
        return self.message

    def show(self, file: t.Optional[t.IO[t.Any]]=None) -> None:
        if file is None:
            file = get_text_stderr()
        echo(_('Error: {message}').format(message=self.format_message()), file=file)