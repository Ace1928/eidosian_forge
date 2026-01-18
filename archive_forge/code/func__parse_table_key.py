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
def _parse_table_key(self, table_key: str) -> Tuple[Optional[str], str]:
    if '.' in table_key:
        tokens = table_key.split('.')
        sname: Optional[str] = '.'.join(tokens[0:-1])
        tname = tokens[-1]
    else:
        tname = table_key
        sname = None
    return (sname, tname)