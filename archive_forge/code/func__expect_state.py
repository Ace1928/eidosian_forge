from __future__ import annotations
import contextlib
from enum import Enum
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from .. import exc as sa_exc
from .. import util
from ..util.typing import Literal
@contextlib.contextmanager
def _expect_state(self, expected: _StateChangeState) -> Iterator[Any]:
    """called within a method that changes states.

        method must also use the ``@declare_states()`` decorator.

        """
    assert self._next_state is _StateChangeStates.CHANGE_IN_PROGRESS, 'Unexpected call to _expect_state outside of state-changing method'
    self._next_state = expected
    try:
        yield
    except:
        raise
    else:
        if self._state is not expected:
            raise sa_exc.IllegalStateChangeError(f'Unexpected state change to {self._state!r}', code='isce')
    finally:
        self._next_state = _StateChangeStates.CHANGE_IN_PROGRESS