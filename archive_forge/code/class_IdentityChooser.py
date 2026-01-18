from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import event
from .. import exc
from .. import inspect
from .. import util
from ..orm import PassiveFlag
from ..orm._typing import OrmExecuteOptionsParameter
from ..orm.interfaces import ORMOption
from ..orm.mapper import Mapper
from ..orm.query import Query
from ..orm.session import _BindArguments
from ..orm.session import _PKIdentityArgument
from ..orm.session import Session
from ..util.typing import Protocol
from ..util.typing import Self
class IdentityChooser(Protocol):

    def __call__(self, mapper: Mapper[_T], primary_key: _PKIdentityArgument, *, lazy_loaded_from: Optional[InstanceState[Any]], execution_options: OrmExecuteOptionsParameter, bind_arguments: _BindArguments, **kw: Any) -> Any:
        ...