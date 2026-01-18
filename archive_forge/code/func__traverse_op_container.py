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
@_traverse.dispatch_for(ops.OpContainer)
def _traverse_op_container(self, context: MigrationContext, revision: _GetRevArg, directive: OpContainer) -> None:
    self._traverse_list(context, revision, directive.ops)