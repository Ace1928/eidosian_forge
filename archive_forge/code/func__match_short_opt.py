import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
def _match_short_opt(self, arg: str, state: ParsingState) -> None:
    stop = False
    i = 1
    prefix = arg[0]
    unknown_options = []
    for ch in arg[1:]:
        opt = normalize_opt(f'{prefix}{ch}', self.ctx)
        option = self._short_opt.get(opt)
        i += 1
        if not option:
            if self.ignore_unknown_options:
                unknown_options.append(ch)
                continue
            raise NoSuchOption(opt, ctx=self.ctx)
        if option.takes_value:
            if i < len(arg):
                state.rargs.insert(0, arg[i:])
                stop = True
            value = self._get_value_from_state(opt, option, state)
        else:
            value = None
        option.process(value, state)
        if stop:
            break
    if self.ignore_unknown_options and unknown_options:
        state.largs.append(f'{prefix}{''.join(unknown_options)}')