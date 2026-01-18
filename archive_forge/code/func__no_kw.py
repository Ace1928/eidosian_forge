from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import roles
from .. import exc
from .. import util
from ..inspection import Inspectable
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypeAlias
def _no_kw() -> exc.ArgumentError:
    return exc.ArgumentError('Additional keyword arguments are not accepted by this function/method.  The presence of **kw is for pep-484 typing purposes')