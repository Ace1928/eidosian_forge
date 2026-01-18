import typing as t
from gettext import gettext as _
from gettext import ngettext
from ._compat import get_text_stderr
from .utils import echo
from .utils import format_filename
class BadParameter(UsageError):
    """An exception that formats out a standardized error message for a
    bad parameter.  This is useful when thrown from a callback or type as
    Click will attach contextual information to it (for instance, which
    parameter it is).

    .. versionadded:: 2.0

    :param param: the parameter object that caused this error.  This can
                  be left out, and Click will attach this info itself
                  if possible.
    :param param_hint: a string that shows up as parameter name.  This
                       can be used as alternative to `param` in cases
                       where custom validation should happen.  If it is
                       a string it's used as such, if it's a list then
                       each item is quoted and separated.
    """

    def __init__(self, message: str, ctx: t.Optional['Context']=None, param: t.Optional['Parameter']=None, param_hint: t.Optional[str]=None) -> None:
        super().__init__(message, ctx)
        self.param = param
        self.param_hint = param_hint

    def format_message(self) -> str:
        if self.param_hint is not None:
            param_hint = self.param_hint
        elif self.param is not None:
            param_hint = self.param.get_error_hint(self.ctx)
        else:
            return _('Invalid value: {message}').format(message=self.message)
        return _('Invalid value for {param_hint}: {message}').format(param_hint=_join_param_hints(param_hint), message=self.message)