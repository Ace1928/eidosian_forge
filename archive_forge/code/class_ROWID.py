from __future__ import annotations
import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR
class ROWID(sqltypes.TypeEngine):
    """Oracle ROWID type.

    When used in a cast() or similar, generates ROWID.

    """
    __visit_name__ = 'ROWID'