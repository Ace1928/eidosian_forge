from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
import weakref
from . import util as orm_util
from .. import exc as sa_exc
def _killed(state: InstanceState[Any], key: _IdentityKeyType[Any]) -> NoReturn:
    raise sa_exc.InvalidRequestError("Object %s cannot be converted to 'persistent' state, as this identity map is no longer valid.  Has the owning Session been closed?" % orm_util.state_str(state), code='lkrp')