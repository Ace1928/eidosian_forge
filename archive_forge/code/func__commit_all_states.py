from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import base
from . import exc as orm_exc
from . import interfaces
from ._typing import _O
from ._typing import is_collection_impl
from .base import ATTR_WAS_SET
from .base import INIT_OK
from .base import LoaderCallableStatus
from .base import NEVER_SET
from .base import NO_VALUE
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import SQL_OK
from .path_registry import PathRegistry
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
@classmethod
def _commit_all_states(self, iter_: Iterable[Tuple[InstanceState[Any], _InstanceDict]], instance_dict: Optional[IdentityMap]=None) -> None:
    """Mass / highly inlined version of commit_all()."""
    for state, dict_ in iter_:
        state_dict = state.__dict__
        state.committed_state.clear()
        if '_pending_mutations' in state_dict:
            del state_dict['_pending_mutations']
        state.expired_attributes.difference_update(dict_)
        if instance_dict and state.modified:
            instance_dict._modified.discard(state)
        state.modified = state.expired = False
        state._strong_obj = None