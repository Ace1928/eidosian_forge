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
def _traverse_list(self, context: MigrationContext, revision: _GetRevArg, directives: Any) -> None:
    dest = []
    for directive in directives:
        dest.extend(self._traverse_for(context, revision, directive))
    directives[:] = dest