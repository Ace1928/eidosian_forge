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
@util.decorator
def _go(fn: _F, self: Any, *arg: Any, **kw: Any) -> Any:
    current_state = self._state
    if has_prerequisite_states and current_state not in prerequisite_state_collection:
        self._raise_for_prerequisite_state(fn.__name__, current_state)
    next_state = self._next_state
    existing_fn = self._current_fn
    expect_state = moves_to if expect_state_change else current_state
    if next_state is not _StateChangeStates.ANY and expect_state_change and (next_state is not expect_state):
        if existing_fn and next_state in (_StateChangeStates.NO_CHANGE, _StateChangeStates.CHANGE_IN_PROGRESS):
            raise sa_exc.IllegalStateChangeError(f"Method '{fn.__name__}()' can't be called here; method '{existing_fn.__name__}()' is already in progress and this would cause an unexpected state change to {moves_to!r}", code='isce')
        else:
            raise sa_exc.IllegalStateChangeError(f"Cant run operation '{fn.__name__}()' here; will move to state {moves_to!r} where we are expecting {next_state!r}", code='isce')
    self._current_fn = fn
    self._next_state = _StateChangeStates.CHANGE_IN_PROGRESS
    try:
        ret_value = fn(self, *arg, **kw)
    except:
        raise
    else:
        if self._state is expect_state:
            return ret_value
        if self._state is current_state:
            raise sa_exc.IllegalStateChangeError(f"Method '{fn.__name__}()' failed to change state to {moves_to!r} as expected", code='isce')
        elif existing_fn:
            raise sa_exc.IllegalStateChangeError(f"While method '{existing_fn.__name__}()' was running, method '{fn.__name__}()' caused an unexpected state change to {self._state!r}", code='isce')
        else:
            raise sa_exc.IllegalStateChangeError(f"Method '{fn.__name__}()' caused an unexpected state change to {self._state!r}", code='isce')
    finally:
        self._next_state = next_state
        self._current_fn = existing_fn