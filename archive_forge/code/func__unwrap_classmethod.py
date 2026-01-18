from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import attributes
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.attributes import QueryableAttribute
from ..sql import roles
from ..sql._typing import is_has_clause_element
from ..sql.elements import ColumnElement
from ..sql.elements import SQLCoreOperations
from ..util.typing import Concatenate
from ..util.typing import Literal
from ..util.typing import ParamSpec
from ..util.typing import Protocol
from ..util.typing import Self
def _unwrap_classmethod(meth: _T) -> _T:
    if isinstance(meth, classmethod):
        return meth.__func__
    else:
        return meth