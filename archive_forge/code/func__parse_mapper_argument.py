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
def _parse_mapper_argument(arg: Union[Mapper[_O], Type[_O]]) -> Mapper[_O]:
    insp = inspection.inspect(arg, raiseerr=False)
    if insp_is_mapper(insp):
        return insp
    raise sa_exc.ArgumentError(f'Mapper or mapped class expected, got {arg!r}')