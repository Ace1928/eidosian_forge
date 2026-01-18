from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import schema as sa_schema
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.schema import Constraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.types import Integer
from sqlalchemy.types import NULLTYPE
from .. import util
from ..util import sqla_compat
def _ensure_table_for_fk(self, metadata: MetaData, fk: ForeignKey) -> None:
    """create a placeholder Table object for the referent of a
        ForeignKey.

        """
    if isinstance(fk._colspec, str):
        table_key, cname = fk._colspec.rsplit('.', 1)
        sname, tname = self._parse_table_key(table_key)
        if table_key not in metadata.tables:
            rel_t = sa_schema.Table(tname, metadata, schema=sname)
        else:
            rel_t = metadata.tables[table_key]
        if cname not in rel_t.c:
            rel_t.append_column(sa_schema.Column(cname, NULLTYPE))