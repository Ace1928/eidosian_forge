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
class concat(GenericFunction[str]):
    """The SQL CONCAT() function, which concatenates strings.

    E.g.:

    .. sourcecode:: pycon+sql

        >>> print(select(func.concat('a', 'b')))
        {printsql}SELECT concat(:concat_2, :concat_3) AS concat_1

    String concatenation in SQLAlchemy is more commonly available using the
    Python ``+`` operator with string datatypes, which will render a
    backend-specific concatenation operator, such as :

    .. sourcecode:: pycon+sql

        >>> print(select(literal("a") + "b"))
        {printsql}SELECT :param_1 || :param_2 AS anon_1


    """
    type = sqltypes.String()
    inherit_cache = True