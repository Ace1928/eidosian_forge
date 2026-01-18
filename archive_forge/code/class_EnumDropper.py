from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from ... import schema
from ... import util
from ...sql import coercions
from ...sql import elements
from ...sql import roles
from ...sql import sqltypes
from ...sql import type_api
from ...sql.base import _NoArg
from ...sql.ddl import InvokeCreateDDLBase
from ...sql.ddl import InvokeDropDDLBase
class EnumDropper(NamedTypeDropper):

    def visit_enum(self, enum):
        if not self._can_drop_type(enum):
            return
        with self.with_ddl_events(enum):
            self.connection.execute(DropEnumType(enum))