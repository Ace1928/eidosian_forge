import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
class ParsingState:

    def __init__(self, rargs: t.List[str]) -> None:
        self.opts: t.Dict[str, t.Any] = {}
        self.largs: t.List[str] = []
        self.rargs = rargs
        self.order: t.List['CoreParameter'] = []