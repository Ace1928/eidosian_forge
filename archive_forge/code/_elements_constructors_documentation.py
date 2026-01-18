from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple as typing_Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from .base import _NoArg
from .coercions import _document_text_coercion
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import Case
from .elements import Cast
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Extract
from .elements import False_
from .elements import FunctionFilter
from .elements import Label
from .elements import Null
from .elements import Over
from .elements import TextClause
from .elements import True_
from .elements import TryCast
from .elements import Tuple
from .elements import TypeCoerce
from .elements import UnaryExpression
from .elements import WithinGroup
from .functions import FunctionElement
from ..util.typing import Literal
Produce a conjunction of expressions joined by ``OR``.

        E.g.::

            from sqlalchemy import or_

            stmt = select(users_table).where(
                            or_(
                                users_table.c.name == 'wendy',
                                users_table.c.name == 'jack'
                            )
                        )

        The :func:`.or_` conjunction is also available using the
        Python ``|`` operator (though note that compound expressions
        need to be parenthesized in order to function with Python
        operator precedence behavior)::

            stmt = select(users_table).where(
                            (users_table.c.name == 'wendy') |
                            (users_table.c.name == 'jack')
                        )

        The :func:`.or_` construct must be given at least one positional
        argument in order to be valid; a :func:`.or_` construct with no
        arguments is ambiguous.   To produce an "empty" or dynamically
        generated :func:`.or_`  expression, from a given list of expressions,
        a "default" element of :func:`_sql.false` (or just ``False``) should be
        specified::

            from sqlalchemy import false
            or_criteria = or_(false(), *expressions)

        The above expression will compile to SQL as the expression ``false``
        or ``0 = 1``, depending on backend, if no other expressions are
        present.  If expressions are present, then the :func:`_sql.false` value
        is ignored as it does not affect the outcome of an OR expression which
        has other elements.

        .. deprecated:: 1.4  The :func:`.or_` element now requires that at
           least one argument is passed; creating the :func:`.or_` construct
           with no arguments is deprecated, and will emit a deprecation warning
           while continuing to produce a blank SQL string.

        .. seealso::

            :func:`.and_`

        