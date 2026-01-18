from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
from .. import exc
from .. import util
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Protocol
from ..util.typing import Self
@classmethod
def _trans_ctx_check(cls, subject: _TConsSubject) -> None:
    trans_context = subject._trans_context_manager
    if trans_context:
        if not trans_context._transaction_is_active():
            raise exc.InvalidRequestError("Can't operate on closed transaction inside context manager.  Please complete the context manager before emitting further commands.")