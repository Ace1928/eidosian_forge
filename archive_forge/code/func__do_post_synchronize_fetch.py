from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
@classmethod
def _do_post_synchronize_fetch(cls, session, statement, result, update_options):
    target_mapper = update_options._subject_mapper
    returned_defaults_rows = result.returned_defaults_rows
    if returned_defaults_rows:
        pk_rows = cls._interpret_returning_rows(target_mapper, returned_defaults_rows)
        matched_rows = [tuple(row) + (update_options._identity_token,) for row in pk_rows]
    else:
        matched_rows = update_options._matched_rows
    for row in matched_rows:
        primary_key = row[0:-1]
        identity_token = row[-1]
        identity_key = target_mapper.identity_key_from_primary_key(list(primary_key), identity_token=identity_token)
        if identity_key in session.identity_map:
            session._remove_newly_deleted([attributes.instance_state(session.identity_map[identity_key])])