import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
def _process_opts(self, arg: str, state: ParsingState) -> None:
    explicit_value = None
    if '=' in arg:
        long_opt, explicit_value = arg.split('=', 1)
    else:
        long_opt = arg
    norm_long_opt = normalize_opt(long_opt, self.ctx)
    try:
        self._match_long_opt(norm_long_opt, explicit_value, state)
    except NoSuchOption:
        if arg[:2] not in self._opt_prefixes:
            self._match_short_opt(arg, state)
            return
        if not self.ignore_unknown_options:
            raise
        state.largs.append(arg)