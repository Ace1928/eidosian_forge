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
def _commit_all(self, dict_: _InstanceDict, instance_dict: Optional[IdentityMap]=None) -> None:
    """commit all attributes unconditionally.

        This is used after a flush() or a full load/refresh
        to remove all pending state from the instance.

         - all attributes are marked as "committed"
         - the "strong dirty reference" is removed
         - the "modified" flag is set to False
         - any "expired" markers for scalar attributes loaded are removed.
         - lazy load callables for objects / collections *stay*

        Attributes marked as "expired" can potentially remain
        "expired" after this step if a value was not populated in state.dict.

        """
    self._commit_all_states([(self, dict_)], instance_dict)