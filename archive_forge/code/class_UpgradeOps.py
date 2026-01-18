from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
class UpgradeOps(OpContainer):
    """contains a sequence of operations that would apply to the
    'upgrade' stream of a script.

    .. seealso::

        :ref:`customizing_revision`

    """

    def __init__(self, ops: Sequence[MigrateOperation]=(), upgrade_token: str='upgrades') -> None:
        super().__init__(ops=ops)
        self.upgrade_token = upgrade_token

    def reverse_into(self, downgrade_ops: DowngradeOps) -> DowngradeOps:
        downgrade_ops.ops[:] = list(reversed([op.reverse() for op in self.ops]))
        return downgrade_ops

    def reverse(self) -> DowngradeOps:
        return self.reverse_into(DowngradeOps(ops=[]))