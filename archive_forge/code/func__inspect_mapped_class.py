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
def _inspect_mapped_class(class_: Type[_O], configure: bool=False) -> Optional[Mapper[_O]]:
    try:
        class_manager = opt_manager_of_class(class_)
        if class_manager is None or not class_manager.is_mapped:
            return None
        mapper = class_manager.mapper
    except exc.NO_STATE:
        return None
    else:
        if configure:
            mapper._check_configure()
        return mapper