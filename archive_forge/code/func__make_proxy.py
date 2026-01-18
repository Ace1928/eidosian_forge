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
def _make_proxy(self, selectable: FromClause, name: Optional[str]=None, key: Optional[str]=None, name_is_truncatable: bool=False, compound_select_cols: Optional[_typing_Sequence[ColumnElement[Any]]]=None, **kw: Any) -> Tuple[str, ColumnClause[_T]]:
    """Create a *proxy* for this column.

        This is a copy of this ``Column`` referenced by a different parent
        (such as an alias or select statement).  The column should
        be used only in select scenarios, as its full DDL/default
        information is not transferred.

        """
    fk = [ForeignKey(col if col is not None else f._colspec, _unresolvable=col is None, _constraint=f.constraint) for f, col in [(fk, fk._resolve_column(raiseerr=False)) for fk in self.foreign_keys]]
    if name is None and self.name is None:
        raise exc.InvalidRequestError("Cannot initialize a sub-selectable with this Column object until its 'name' has been assigned.")
    try:
        c = self._constructor(coercions.expect(roles.TruncatedLabelRole, name if name else self.name) if name_is_truncatable else name or self.name, self.type, *fk, key=key if key else name if name else self.key, primary_key=self.primary_key, nullable=self.nullable, _proxies=list(compound_select_cols) if compound_select_cols else [self])
    except TypeError as err:
        raise TypeError('Could not create a copy of this %r object.  Ensure the class includes a _constructor() attribute or method which accepts the standard Column constructor arguments, or references the Column class itself.' % self.__class__) from err
    c.table = selectable
    c._propagate_attrs = selectable._propagate_attrs
    if selectable._is_clone_of is not None:
        c._is_clone_of = selectable._is_clone_of.columns.get(c.key)
    if self.primary_key:
        selectable.primary_key.add(c)
    if fk:
        selectable.foreign_keys.update(fk)
    return (c.key, c)