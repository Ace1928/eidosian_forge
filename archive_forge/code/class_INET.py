from __future__ import annotations
import datetime as dt
from typing import Any
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from ...sql import sqltypes
from ...sql import type_api
from ...util.typing import Literal
class INET(sqltypes.TypeEngine[str]):
    __visit_name__ = 'INET'