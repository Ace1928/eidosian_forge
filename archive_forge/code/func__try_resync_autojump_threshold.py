import time
from math import inf
from .. import _core
from .._abc import Clock
from .._util import final
from ._run import GLOBAL_RUN_CONTEXT
def _try_resync_autojump_threshold(self) -> None:
    try:
        runner = GLOBAL_RUN_CONTEXT.runner
        if runner.is_guest:
            runner.force_guest_tick_asap()
    except AttributeError:
        pass
    else:
        runner.clock_autojump_threshold = self._autojump_threshold