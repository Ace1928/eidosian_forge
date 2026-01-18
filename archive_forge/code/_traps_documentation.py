from __future__ import annotations
import enum
import types
from typing import TYPE_CHECKING, Any, Callable, NoReturn
import attrs
import outcome
from . import _run
Reattach a coroutine object that was detached using
    :func:`temporarily_detach_coroutine_object`.

    When the calling coroutine enters this function it's running under the
    foreign coroutine runner, and when the function returns it's running under
    Trio.

    This must be called from inside the coroutine being resumed, and yields
    whatever value you pass in. (Presumably you'll pass a value that will
    cause the current coroutine runner to stop scheduling this task.) Then the
    coroutine is resumed by the Trio scheduler at the next opportunity.

    Args:
      task (Task): The Trio task object that the current coroutine was
          detached from.
      yield_value (object): The object to yield to the current coroutine
          runner.

    