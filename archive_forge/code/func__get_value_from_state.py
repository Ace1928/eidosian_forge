import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
def _get_value_from_state(self, option_name: str, option: Option, state: ParsingState) -> t.Any:
    nargs = option.nargs
    if len(state.rargs) < nargs:
        if option.obj._flag_needs_value:
            value = _flag_needs_value
        else:
            raise BadOptionUsage(option_name, ngettext('Option {name!r} requires an argument.', 'Option {name!r} requires {nargs} arguments.', nargs).format(name=option_name, nargs=nargs))
    elif nargs == 1:
        next_rarg = state.rargs[0]
        if option.obj._flag_needs_value and isinstance(next_rarg, str) and (next_rarg[:1] in self._opt_prefixes) and (len(next_rarg) > 1):
            value = _flag_needs_value
        else:
            value = state.rargs.pop(0)
    else:
        value = tuple(state.rargs[:nargs])
        del state.rargs[:nargs]
    return value