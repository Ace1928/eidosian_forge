from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
def _resolve_col_tokens(self) -> Tuple[Table, str, Optional[str]]:
    if self.parent is None:
        raise exc.InvalidRequestError('this ForeignKey object does not yet have a parent Column associated with it.')
    elif self.parent.table is None:
        raise exc.InvalidRequestError("this ForeignKey's parent column is not yet associated with a Table.")
    parenttable = self.parent.table
    if self._unresolvable:
        schema, tname, colname = self._column_tokens
        tablekey = _get_table_key(tname, schema)
        return (parenttable, tablekey, colname)
    for c in self.parent.base_columns:
        if isinstance(c, Column):
            assert c.table is parenttable
            break
    else:
        assert False
    schema, tname, colname = self._column_tokens
    if schema is None and parenttable.metadata.schema is not None:
        schema = parenttable.metadata.schema
    tablekey = _get_table_key(tname, schema)
    return (parenttable, tablekey, colname)