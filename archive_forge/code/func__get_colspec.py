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
def _get_colspec(self, schema: Optional[Union[str, Literal[SchemaConst.RETAIN_SCHEMA, SchemaConst.BLANK_SCHEMA]]]=None, table_name: Optional[str]=None, _is_copy: bool=False) -> str:
    """Return a string based 'column specification' for this
        :class:`_schema.ForeignKey`.

        This is usually the equivalent of the string-based "tablename.colname"
        argument first passed to the object's constructor.

        """
    if schema not in (None, RETAIN_SCHEMA):
        _schema, tname, colname = self._column_tokens
        if table_name is not None:
            tname = table_name
        if schema is BLANK_SCHEMA:
            return '%s.%s' % (tname, colname)
        else:
            return '%s.%s.%s' % (schema, tname, colname)
    elif table_name:
        schema, tname, colname = self._column_tokens
        if schema:
            return '%s.%s.%s' % (schema, table_name, colname)
        else:
            return '%s.%s' % (table_name, colname)
    elif self._table_column is not None:
        if self._table_column.table is None:
            if _is_copy:
                raise exc.InvalidRequestError(f"Can't copy ForeignKey object which refers to non-table bound Column {self._table_column!r}")
            else:
                return self._table_column.key
        return '%s.%s' % (self._table_column.table.fullname, self._table_column.key)
    else:
        assert isinstance(self._colspec, str)
        return self._colspec