from __future__ import annotations
import operator
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from ..engine.interfaces import _CoreKnownExecutionOptions
from ..sql import roles
from ..sql._orm_types import DMLStrategyArgument as DMLStrategyArgument
from ..sql._orm_types import (
from ..sql._typing import _HasClauseElement
from ..sql.elements import ColumnElement
from ..util.typing import Protocol
from ..util.typing import TypeGuard
class _OrmKnownExecutionOptions(_CoreKnownExecutionOptions, total=False):
    populate_existing: bool
    autoflush: bool
    synchronize_session: SynchronizeSessionArgument
    dml_strategy: DMLStrategyArgument
    is_delete_using: bool
    is_update_from: bool
    render_nulls: bool