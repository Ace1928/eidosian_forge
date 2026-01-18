import time
from math import inf
from .. import _core
from .._abc import Clock
from .._util import final
from ._run import GLOBAL_RUN_CONTEXT
Manually advance the clock by the given number of seconds.

        Args:
          seconds (float): the number of seconds to jump the clock forward.

        Raises:
          ValueError: if you try to pass a negative value for ``seconds``.

        