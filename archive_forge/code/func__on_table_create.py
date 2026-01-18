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
def _on_table_create(self, target, bind, checkfirst=False, **kw):
    if (checkfirst or (not self.metadata and (not kw.get('_is_metadata_operation', False)))) and (not self._check_for_name_in_memos(checkfirst, kw)):
        self.create(bind=bind, checkfirst=checkfirst)