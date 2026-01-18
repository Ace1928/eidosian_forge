from __future__ import annotations
import datetime
import decimal
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import annotation
from . import coercions
from . import operators
from . import roles
from . import schema
from . import sqltypes
from . import type_api
from . import util as sqlutil
from ._typing import is_table_value_type
from .base import _entity_namespace
from .base import ColumnCollection
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .elements import _type_from_args
from .elements import BinaryExpression
from .elements import BindParameter
from .elements import Cast
from .elements import ClauseList
from .elements import ColumnElement
from .elements import Extract
from .elements import FunctionFilter
from .elements import Grouping
from .elements import literal_column
from .elements import NamedColumn
from .elements import Over
from .elements import WithinGroup
from .selectable import FromClause
from .selectable import Select
from .selectable import TableValuedAlias
from .sqltypes import TableValueType
from .type_api import TypeEngine
from .visitors import InternalTraversal
from .. import util
class next_value(GenericFunction[int]):
    """Represent the 'next value', given a :class:`.Sequence`
    as its single argument.

    Compiles into the appropriate function on each backend,
    or will raise NotImplementedError if used on a backend
    that does not provide support for sequences.

    """
    type = sqltypes.Integer()
    name = 'next_value'
    _traverse_internals = [('sequence', InternalTraversal.dp_named_ddl_element)]

    def __init__(self, seq: schema.Sequence, **kw: Any):
        assert isinstance(seq, schema.Sequence), 'next_value() accepts a Sequence object as input.'
        self.sequence = seq
        self.type = sqltypes.to_instance(seq.data_type or getattr(self, 'type', None))

    def compare(self, other: Any, **kw: Any) -> bool:
        return isinstance(other, next_value) and self.sequence.name == other.sequence.name

    @property
    def _from_objects(self) -> Any:
        return []