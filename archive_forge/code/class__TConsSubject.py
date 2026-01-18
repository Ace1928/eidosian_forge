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
class _TConsSubject(Protocol):
    _trans_context_manager: Optional[TransactionalContext]