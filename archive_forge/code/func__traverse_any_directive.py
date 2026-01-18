from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from .. import util
from ..operations import ops
@_traverse.dispatch_for(ops.MigrateOperation)
def _traverse_any_directive(self, context: MigrationContext, revision: _GetRevArg, directive: MigrateOperation) -> None:
    pass