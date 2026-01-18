from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
class LoaderCallableStatus(Enum):
    PASSIVE_NO_RESULT = 0
    'Symbol returned by a loader callable or other attribute/history\n    retrieval operation when a value could not be determined, based\n    on loader callable flags.\n    '
    PASSIVE_CLASS_MISMATCH = 1
    'Symbol indicating that an object is locally present for a given\n    primary key identity but it is not of the requested class.  The\n    return value is therefore None and no SQL should be emitted.'
    ATTR_WAS_SET = 2
    'Symbol returned by a loader callable to indicate the\n    retrieved value, or values, were assigned to their attributes\n    on the target object.\n    '
    ATTR_EMPTY = 3
    'Symbol used internally to indicate an attribute had no callable.'
    NO_VALUE = 4
    "Symbol which may be placed as the 'previous' value of an attribute,\n    indicating no value was loaded for an attribute when it was modified,\n    and flags indicated we were not to load it.\n    "
    NEVER_SET = NO_VALUE
    '\n    Synonymous with NO_VALUE\n\n    .. versionchanged:: 1.4   NEVER_SET was merged with NO_VALUE\n\n    '